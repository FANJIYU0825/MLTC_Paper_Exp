"""
noise_filter
============
多標籤雜訊偵測與標籤校正套件。

模組結構（依賴由上至下）
------------------------
gmm_core.py    — GMMFitter, apply_epsilon_band          (無內部依賴)
normalize.py   — sanitize_array, normalize, ...         (無內部依賴)
calculators.py — BaseScoreCalculator 及所有子類別       (依賴 normalize)
filters.py     — BaseNoiseFilter, GMMNoiseFilter,       (依賴 gmm_core)
                 IntersectionGMM3Filter, StageOneRELFlipper
correctors.py  — LabelRefiner, RELOnlyCorrector,        (依賴 gmm_core)
                 TwoStageRELFPCorrector,
                 FrequencyAware1DGMMCorrector
pipeline.py    — HSMHybridPipeline, TwoStagePipeline    (依賴以上全部)

快速匯入
--------
from noise_filter import HSMHybridPipeline, TwoStagePipeline
from noise_filter import LabelRefiner, TwoStageRELFPCorrector
from noise_filter import GMMNoiseFilter
from noise_filter import normalize, GMMFitter
"""

from .gmm_core import GMMFitter, apply_epsilon_band
from .normalize import normalize, sanitize_array

from .calculators import (
    BaseScoreCalculator,
    StandardLossCalculator,
    StandardLossPerCellCalculator,
    RankWeightedLossCalculator,
    PrototypeDistanceCalculator,
    PositiveGapCalculator,
)

from .filters import (
    BaseNoiseFilter,
    GMMNoiseFilter,
    IntersectionGMM3Filter,
    StageOneRELFlipper,
)

from .correctors import (
    LabelRefiner,
    RELOnlyCorrector,
    TwoStageRELFPCorrector,
    FrequencyAware1DGMMCorrector,
)

from .pipeline import (
    HSMHybridPipeline,
    TwoStagePipeline,
)

__all__ = [
    # gmm_core
    "GMMFitter",
    "apply_epsilon_band",
    # normalize
    "normalize",
    "sanitize_array",
    # calculators
    "BaseScoreCalculator",
    "StandardLossCalculator",
    "StandardLossPerCellCalculator",
    "RankWeightedLossCalculator",
    "PrototypeDistanceCalculator",
    "PositiveGapCalculator",
    # filters
    "BaseNoiseFilter",
    "GMMNoiseFilter",
    "IntersectionGMM3Filter",
    "StageOneRELFlipper",
    # correctors
    "LabelRefiner",
    "RELOnlyCorrector",
    "TwoStageRELFPCorrector",
    "FrequencyAware1DGMMCorrector",
    # pipeline
    "HSMHybridPipeline",
    "TwoStagePipeline",
]