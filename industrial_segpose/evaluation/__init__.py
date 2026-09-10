"""算法评估子包：集中管理基准测试、测试套件和结果可视化。"""

from .benchmark import EvaluationThresholds, SyntheticStressConfig, run_synthetic_benchmark
from .real_dataset import RealDatasetValidationResult, run_real_dataset_validation

__all__ = [
    "EvaluationThresholds",
    "RealDatasetValidationResult",
    "SyntheticStressConfig",
    "run_real_dataset_validation",
    "run_synthetic_benchmark",
]
