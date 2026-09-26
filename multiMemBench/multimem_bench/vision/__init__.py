"""Optional vision preprocessing pipeline.

The evaluator itself consumes JSON artifacts and stays lightweight. Modules in
this package build those artifacts from real images/videos when optional vision
dependencies such as SAM3 and DINOv3 are installed.
"""

from multimem_bench.vision.config import VisionConfig

__all__ = ["VisionConfig"]
