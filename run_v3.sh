#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH="$ROOT/multiMemBench${PYTHONPATH:+:$PYTHONPATH}"

exec "${PYTHON:-python3}" -m multimem_bench.cli run-benchmark_v3 "$@"
