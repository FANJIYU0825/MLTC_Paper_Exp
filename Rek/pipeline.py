"""
pipeline.py
===========
流程協調器（Orchestrator）。

各個 Pipeline 負責把 calculators / correctors / filters / normalize
串接成完整的雜訊偵測與校正流程，本身不含任何演算法邏輯。

公開 API
--------
HSMHybridPipeline   ← REL + CD (+ Gap) 分數融合，可接 run_score_only /
                      run_score_separately
TwoStagePipeline    ← Stage1 REL flip + Stage2 CD∩Gap GMM(3)

依賴
----
calculators.py — BaseScoreCalculator 子類別
correctors.py  — （由呼叫端注入）
filters.py     — StageOneRELFlipper, IntersectionGMM3Filter
normalize.py   — normalize
"""

from __future__ import annotations

import numpy as np

from .normalize import normalize as _normalize_fn
from .filters import StageOneRELFlipper, IntersectionGMM3Filter

try:
    from util.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def _save_scores(path: str, **arrays) -> None:
    """將多個 numpy 陣列存成 .npz.compressed。"""
    np.savez_compressed(path, **arrays)
    logger.info(f"Scores saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. HSMHybridPipeline
# ─────────────────────────────────────────────────────────────────────────────

class HSMHybridPipeline:
    """
    HSM 分數融合流程。

    融合公式
    --------
    HSM = α × REL_norm + (1−α) × [β × CD_norm + (1−β) × Gap_norm]

    β=1.0 → 純 CD（退化為原始 HSM，不使用 Gap）
    β=0.0 → 純 Gap（不使用 CD）

    Parameters
    ----------
    rel_calculator : BaseScoreCalculator
        計算 REL 分數的策略物件（如 RankWeightedLossCalculator）。
    cd_calculator  : BaseScoreCalculator
        計算 CD 分數的策略物件（如 PrototypeDistanceCalculator）。
    alpha          : float   REL 的融合權重，預設 0.5
    gap_calculator : BaseScoreCalculator | None
        計算 Gap 分數的策略物件（如 PositiveGapCalculator），可選。
    beta           : float   CD 在 (1-alpha) 部分的佔比，預設 1.0
    """

    def __init__(
        self,
        rel_calculator,
        cd_calculator,
        alpha:          float = 0.5,
        gap_calculator        = None,
        beta:           float = 1.0,
    ) -> None:
        self.rel_calc  = rel_calculator
        self.cd_calc   = cd_calculator
        self.gap_calc  = gap_calculator
        self.alpha     = alpha
        self.beta      = beta

    # ── 公開方法 ─────────────────────────────────────────────────────────────

    def run_score_only(
        self,
        model,
        dataloader,
        device,
        normalization: str        = "minmax",
        clip_range                = None,
        dataset_name:  str        = "dataset",
        encoder_name:  str        = "mltc",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        計算、標準化並融合分數，不執行任何篩選或校正。
        適合外部參數搜尋使用。

        Returns
        -------
        hsm_scores : [N, C] 融合後分數
        labels     : [N, C] 原始標籤
        indices    : [N]    樣本索引
        """
        logger.info("[HSMPipeline] run_score_only ...")

        idx_r, rel_scores, labels = self.rel_calc.calculate(model, dataloader, device)
        idx_c, cd_scores,  _      = self.cd_calc.calculate(model, dataloader, device)

        norm_kwargs = dict(clip_range=clip_range) if clip_range else {}
        rel_norm    = _normalize_fn(rel_scores, normalization, **norm_kwargs)
        cd_norm     = _normalize_fn(cd_scores,  normalization, **norm_kwargs)

        save_kwargs = dict(
            idx_rel=idx_r, rel_scores=rel_scores, rel_norm=rel_norm,
            idx_cd=idx_c,  cd_scores=cd_scores,   cd_norm=cd_norm,
            labels=labels, normalization=normalization,
            alpha=self.alpha, beta=self.beta,
        )

        if self.gap_calc is not None:
            _, gap_scores, _ = self.gap_calc.calculate(model, dataloader, device)
            gap_norm         = _normalize_fn(gap_scores, normalization, **norm_kwargs)
            save_kwargs.update(gap_scores=gap_scores, gap_norm=gap_norm)
            cd_part          = self.beta * cd_norm + (1 - self.beta) * gap_norm
            logger.info(
                f"  Fusion: α={self.alpha} β={self.beta} (REL + CD + Gap)"
            )
        else:
            cd_part = cd_norm
            logger.info(f"  Fusion: α={self.alpha} (REL + CD, no Gap)")

        suffix    = f"_{normalization}" if normalization != "minmax" else ""
        save_path = (
            f"data/evaluation_results_{dataset_name}_{encoder_name}{suffix}.npz"
        )
        _save_scores(save_path, **save_kwargs)

        hsm_scores = self.alpha * rel_norm + (1 - self.alpha) * cd_part
        return hsm_scores, labels, idx_r

    def run_score_separately(
        self,
        model,
        dataloader,
        device,
        normalization: str = "minmax",
        clip_range         = None,
        dataset_name: str  = "dataset",
        encoder_name: str  = "mltc",
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray,   # raw: rel, cd, gap
        np.ndarray, np.ndarray, np.ndarray,   # norm: rel, cd, gap
        np.ndarray, np.ndarray,               # labels, indices
    ]:
        """
        計算並分別回傳各項原始分數（不融合），供兩階段校正器使用。

        Raises
        ------
        ValueError
            若未傳入 gap_calculator。

        Returns
        -------
        rel_scores, cd_scores, gap_scores  : [N, C] 原始分數
        rel_norm,   cd_norm,   gap_norm    : [N, C] 標準化分數
        labels                             : [N, C]
        indices                            : [N]
        """
        if self.gap_calc is None:
            raise ValueError(
                "run_score_separately 需要 gap_calculator，"
                "請在初始化時傳入 gap_calculator=PositiveGapCalculator(...)。"
            )

        logger.info("[HSMPipeline] run_score_separately ...")

        idx_r, rel_scores, labels = self.rel_calc.calculate(model, dataloader, device)
        idx_c, cd_scores,  _      = self.cd_calc.calculate(model, dataloader, device)
        idx_m, gap_scores, _      = self.gap_calc.calculate(model, dataloader, device)

        norm_kwargs = dict(clip_range=clip_range) if clip_range else {}
        rel_norm    = _normalize_fn(rel_scores, normalization, **norm_kwargs)
        cd_norm     = _normalize_fn(cd_scores,  normalization, **norm_kwargs)
        gap_norm    = _normalize_fn(gap_scores, normalization, **norm_kwargs)

        logger.info(
            f"  [run_score_separately] "
            f"rel={rel_scores.shape} cd={cd_scores.shape} gap={gap_scores.shape}"
        )
        return (
            rel_scores, cd_scores, gap_scores,
            rel_norm,   cd_norm,   gap_norm,
            labels, idx_r,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TwoStagePipeline
# ─────────────────────────────────────────────────────────────────────────────

class TwoStagePipeline:
    """
    兩階段標籤校正流程。

    Stage 1（可選）: REL top-k 直接翻轉（FP only）
    Stage 2（可選）: CD GMM(n) ∩ Gap GMM(n) suspicious cells
        stage2_action='flip'    → 翻轉這些 cells
        stage2_action='discard' → 從 BCE loss 中遮蔽這些 cells
        stage2_action='none'    → 不動

    Parameters
    ----------
    use_stage1     : bool   是否執行 Stage 1，預設 True
    stage2_action  : str    'flip' | 'discard' | 'none'
    top_ratio      : float  Stage 1 翻轉比例，預設 0.01
    n_components   : int    Stage 2 GMM component 數，預設 3

    Returns（run 方法）
    -------------------
    final_labels : [N, C] float32
    loss_mask    : [N, C] float32  (1=參與 BCE loss, 0=遮蔽)
    """

    def __init__(
        self,
        use_stage1:    bool  = True,
        stage2_action: str   = "flip",
        top_ratio:     float = 0.01,
        n_components:  int   = 3,
    ) -> None:
        if stage2_action not in ("flip", "discard", "none"):
            raise ValueError(
                f"stage2_action='{stage2_action}' 不合法，"
                f"請選 'flip'、'discard' 或 'none'。"
            )
        self.use_stage1    = use_stage1
        self.stage2_action = stage2_action
        self.top_ratio     = top_ratio
        self.n_components  = n_components

    def run(
        self,
        rel_scores:   np.ndarray,
        cd_scores:    np.ndarray,
        gap_scores:   np.ndarray,
        noisy_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        rel_scores, cd_scores, gap_scores : [N, C] 已標準化分數
        noisy_labels                       : [N, C] 原始 noisy 標籤（0/1）

        Returns
        -------
        final_labels : [N, C] float32
        loss_mask    : [N, C] float32
        """
        labels    = np.asarray(noisy_labels, dtype=np.float32).copy()
        N, C      = labels.shape
        loss_mask = np.ones((N, C), dtype=np.float32)

        # ── Stage 1 ──────────────────────────────────────────────────────────
        if self.use_stage1:
            flipper        = StageOneRELFlipper(top_ratio=self.top_ratio)
            labels, _      = flipper.apply(rel_scores, labels.astype(int))
            labels         = labels.astype(np.float32)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        if self.stage2_action == "none":
            return labels, loss_mask

        gmm_filter = IntersectionGMM3Filter(n_components=self.n_components)
        susp_mask  = gmm_filter.intersect(cd_scores, gap_scores, labels.astype(int))

        if self.stage2_action == "flip":
            labels[susp_mask] = 1.0 - labels[susp_mask]
        elif self.stage2_action == "discard":
            loss_mask[susp_mask] = 0.0

        logger.info(
            f"[TwoStagePipeline] use_stage1={self.use_stage1} "
            f"stage2={self.stage2_action} "
            f"affected={int(susp_mask.sum())} cells"
        )
        return labels, loss_mask