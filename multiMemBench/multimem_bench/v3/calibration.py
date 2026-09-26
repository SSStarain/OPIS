"""Offline case-level statistics; never authorizes a calibrated leaderboard."""
from __future__ import annotations

import math
from typing import Any
import numpy as np


def read_case_records(payload: Any) -> list[dict]:
    rows = payload if isinstance(payload, list) else payload.get('per_case') if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) or 'case_id' not in row for row in rows):
        raise ValueError('per-case records required; aggregate means cannot be bootstrapped')
    return rows


def _case_values(rows: list[dict], metric: str) -> dict[str, float | None]:
    values = {}
    for row in read_case_records(rows):
        identifier = row['case_id']
        if not isinstance(identifier, str) or not identifier or identifier in values:
            raise ValueError('case_id must be nonempty and unique within each model')
        value = row.get('metrics', row).get(metric)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{metric} must be finite numeric or null')
            value = float(value)
        values[identifier] = value
    return values


def _bootstrap(values: list[float], samples: int, seed: int) -> dict:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError('samples must be a positive integer')
    if not values:
        return {'estimate': None, 'ci95': None, 'n_valid': 0, 'samples': samples, 'seed': seed,
                'bootstrap_unit': 'case', 'interval': 'percentile', 'conditional_on': 'nonmissing_cases'}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    # Bounded memory even for large case collections and 10,000+ draws.
    estimates = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(samples)]
    return {'estimate': float(array.mean()), 'ci95': np.quantile(estimates, [.025, .975]).tolist(),
            'n_valid': len(values), 'samples': samples, 'seed': seed,
            'bootstrap_unit': 'case', 'interval': 'percentile', 'conditional_on': 'nonmissing_cases'}


def bootstrap_cases(rows: list[dict], *, metric: str, samples: int = 10000, seed: int = 0) -> dict:
    values = _case_values(rows, metric)
    present = [v for _, v in sorted(values.items()) if v is not None]
    return {**_bootstrap(present, samples, seed), 'metric': metric, 'n_cases': len(values),
            'n_missing': len(values) - len(present), 'estimand': 'case_macro'}


def bootstrap_paired_delta(left: list[dict], right: list[dict], *, metric: str,
                           samples: int = 10000, seed: int = 0) -> dict:
    a, b = _case_values(left, metric), _case_values(right, metric)
    paired = sorted(key for key in a.keys() & b.keys() if a[key] is not None and b[key] is not None)
    differences = [a[key] - b[key] for key in paired]
    return {**_bootstrap(differences, samples, seed), 'metric': metric, 'direction': 'left_minus_right',
            'estimand': 'paired_case_macro_delta', 'case_ids': paired,
            'n_cases': len(a.keys() | b.keys()), 'unmatched_case_ids': sorted(a.keys() ^ b.keys()),
            'missing_pair_case_ids': sorted((a.keys() & b.keys()) - set(paired))}


def validate_splits(rows: list[dict]) -> None:
    cases, families = {}, {}
    for row in rows:
        case, split = row.get('case_id'), row.get('split')
        family = row.get('family_id')
        if not case or not family or split not in {'calibration', 'validation', 'test'}:
            raise ValueError('case_id, family_id and explicit calibration/validation/test split required')
        if case in cases:
            raise ValueError('duplicate case_id in split manifest')
        if family in families and families[family] != split:
            raise ValueError('source family leaks across splits')
        cases[case], families[family] = split, split


def validation_statistics(predicted: list[float], target: list[float]) -> dict:
    a, b = np.asarray(predicted, dtype=float), np.asarray(target, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('equal-length finite vectors required')
    def ranks(values):
        return np.array([1 + (values < value).sum() + ((values == value).sum()-1)/2 for value in values])
    ra, rb = ranks(a), ranks(b)
    spearman = float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 1 and ra.std() > 0 and rb.std() > 0 else None
    concordant = discordant = tie_a = tie_b = target_pairs = 0
    for i in range(len(a)):
        for j in range(i):
            x, y = np.sign(a[i]-a[j]), np.sign(b[i]-b[j])
            target_pairs += y != 0
            concordant += x*y > 0
            discordant += x*y < 0
            tie_a += x == 0 and y != 0
            tie_b += y == 0 and x != 0
    denominator = math.sqrt((concordant+discordant+tie_a)*(concordant+discordant+tie_b))
    return {'n_cases': len(a), 'spearman': spearman,
            'kendall_tau_b': float((concordant-discordant)/denominator) if denominator else None,
            'pairwise_accuracy': float(concordant/target_pairs) if target_pairs else None,
            'pairwise_tie_policy': 'target_ties_excluded_prediction_ties_incorrect'}


def select_candidate(candidates: list[dict], *, split: str = 'calibration') -> dict:
    """Select among precomputed evaluator runs using case-level labeled targets.

    This does not infer labels, rerun a backend, fit human-preference weights,
    freeze a config, or estimate a test-set result.
    """
    from .config import V3EvaluationConfig
    if split != 'calibration':
        raise ValueError('candidate selection only permitted on calibration split')
    if not candidates:
        raise ValueError('at least one candidate is required')
    identifiers, runs, expected = set(), [], None
    targets = {}
    for candidate in candidates:
        identifier = candidate.get('candidate_id')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('candidate_id must be unique and nonempty')
        identifiers.add(identifier)
        V3EvaluationConfig.from_dict(candidate['config'])
        cases = candidate['cases']
        validate_splits(cases)
        if any(row['split'] != split for row in cases):
            raise ValueError('candidate search input must contain calibration cases only')
        indexed = {row['case_id']: row for row in cases}
        if expected is None:
            expected = set(indexed)
        elif expected != set(indexed):
            raise ValueError('candidates must use identical case sets')
        for key, row in indexed.items():
            signature = (row['family_id'], row.get('target'))
            if key in targets and targets[key] != signature:
                raise ValueError('candidate target/family metadata mismatch')
            targets[key] = signature
        runs.append((candidate, indexed))
    valid = []
    for key in sorted(expected):
        eligible = True
        for _, indexed in runs:
            row = indexed[key]
            for name in ('predicted', 'target'):
                value = row.get(name)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError('predicted and target must be finite numeric or null')
            eligible &= (row.get('evaluator_failure') is False and row.get('motion_valid') is True
                         and row.get('coverage_qualified') is True and row.get('predicted') is not None
                         and row.get('target') is not None)
        if eligible:
            valid.append(key)
    if not valid:
        raise ValueError('no common qualified labeled calibration cases')
    results = []
    for candidate, indexed in runs:
        predicted = [indexed[key]['predicted'] for key in valid]
        target = [indexed[key]['target'] for key in valid]
        mse = float(np.mean((np.asarray(predicted)-np.asarray(target))**2))
        results.append({'candidate_id': candidate['candidate_id'], 'mean_squared_error': mse,
                        'statistics': validation_statistics(predicted, target)})
    winner = min(results, key=lambda row: (row['mean_squared_error'], row['candidate_id']))
    selected = next(candidate for candidate, _ in runs if candidate['candidate_id'] == winner['candidate_id'])
    return {'selected_candidate_id': winner['candidate_id'], 'config': selected['config'], 'split': split,
            'n_valid': len(valid), 'case_ids': valid, 'excluded_case_ids': sorted(expected-set(valid)),
            'candidates': results, 'provisional': True, 'release_eligible': False,
            'objective': 'case_mean_squared_error', 'weights_fitted': False,
            'remaining': ['independent_validation', 'human_preference_weight_fit', 'protocol_freeze']}
