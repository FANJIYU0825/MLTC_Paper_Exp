"""
filters.py
==========
樣本 / Cell 層級的篩選工具(filter 語意）。

與 correctors.py 的區別：
    filters.py  → 判斷「哪些樣本 / cell 是雜訊」，回傳 mask 或 index
    correctors.py → 決定「把雜訊標籤改成多少」，回傳校正後的 label 矩陣

公開 API
--------
BaseNoiseFilter          ← 抽象基底(filter 介面）
GMMNoiseFilter           ← 1D GMM 二分法filter() 回傳 clean/noisy indices
IntersectionGMM3Filter   ← per-label GMM(3) 取 CD ∩ Gap suspicious mask
StageOneRELFlipper       ← REL top-k 直接翻轉(Stage 1 helper)

依賴
----
gmm_core.py — GMMFitter
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from .experence import  visualize_gmm
from .gmm_core import GMMFitter, apply_epsilon_band

try:
    from util.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


#─────────────────────────────────────────────────────────────────────────────
#抽象基底
# ─────────────────────────────────────────────────────────────────────────────

class BaseNoiseFilter(ABC):
    """
    樣本篩選的抽象介面。

    filter()接收分數與索引，回傳 (clean_indices, noisy_indices)。
    """

    @abstractmethod
    def filter(
        self,
        scores:np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

#─────────────────────────────────────────────────────────────────────────────
#GMMNoiseFilter
# ─────────────────────────────────────────────────────────────────────────────

class GMMNoiseFilter(BaseNoiseFilter):
    """
    1D GMM 二分法樣本篩選器。

    將輸入的分數(loss / distance)以 GMM 分成兩群，
    mean 較小的群視為乾淨樣本（clean）。

    支援輸入格式
    ------------
    - [N]      : 已聚合的 1D 分數
    - [N, C]   : Multi-label 矩陣，自動 sum over C → [N]
    - [NXC]    : 已 flatten 的矩陣，自動 reshape → sum → [N]

    Parameters
    ----------
    n_components : int   GMM component 數，預設 2
    threshold    : float 判斷為 clean 的後驗機率門檻，預設 0.5
    """

    def __init__(self, n_components: int = 2, threshold: float = 0.5) -> None:
        self.n_components = n_components
        self.threshold    = threshold

    #── 公開方法 ─────────────────────────────────────────────────────────────

    def filter(
        self,
        scores:  np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        以 GMM 二分法篩選樣本。

        Returns
        -------
        clean_indices : np.ndarray  乾淨樣本索引
        noisy_indices : np.ndarray  雜訊樣本索引
        """
        scores  = np.asarray(scores,  dtype=float)
        indices = np.asarray(indices)
        scores  = self._coerce_to_1d(scores, indices)

        X         = scores.reshape(-1, 1)
        gmm       = GMM(
            n_components=self.n_components,
            max_iter=100, tol=1e-2, reg_covar=5e-4,
        )
        gmm.fit(X)

        clean_comp  = int(np.argmin(np.ravel(gmm.means_)))
        prob_clean  = gmm.predict_proba(X)[:, clean_comp]
        is_clean    = prob_clean > self.threshold

        logger.info(
            f"[GMMNoiseFilter] total={len(indices)} "
            f"clean={is_clean.sum()} ({is_clean.mean():.1%}) "
            f"noisy={(~is_clean).sum()} ({(~is_clean).mean():.1%})"
        )
        return indices[is_clean], indices[~is_clean]

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_to_1d(scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        統一把各種輸入格式轉成 [N] 1D 分數。

        處理優先順序：
        1. [N, C] matrix → sum(axis=1) → [N]
        2. 已是 [N] 且長度正確 → 直接使用
        3. [N×C] flattened → reshape → sum → [N]（長度為 indices 的整數倍）
        """
        N = len(indices)

        if scores.ndim == 2 and scores.shape[0] == N:
            return scores.sum(axis=1)

        if scores.ndim == 1 and len(scores) == N:
            return scores

        if scores.ndim == 1 and len(scores) > N and len(scores) % N == 0:
            C = len(scores) // N
            logger.debug(
                f"[GMMNoiseFilter] 偵測到 flatten 的 multi-label 輸入，"
                f"reshape({N}, {C}) 後 sum。"
            )
            return scores.reshape(N, C).sum(axis=1)

        raise ValueError(
            f"[GMMNoiseFilter] 無法對齊分數與索引：scores={scores.shape}, N={N}。"
        )

    def correction_perlabel(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        args,
    ) -> np.ndarray:
        """
        Per-label GMM 校正：對每個 label c，分別對正樣本與負樣本做 GMM，
        以 epsilon-band 產生 soft label。

        Parameters
        ----------
        scores : [N, C] float  已標準化或融合的分數
        labels : [N, C] int    原始 noisy 標籤
        args   : 含 args.epsilon (必要), args.alpha, args.targert_list,
                 args.label_index_path, args.encoder_name (用於視覺化檔名)

        Returns
        -------
        refined_labels : [N, C] float  校正後標籤（可能含 soft 值）
        """
        logger.info("[GMMNoiseFilter] Running Per-Label GMM Correction...")

        scores         = np.nan_to_num(
            np.asarray(scores, dtype=float), nan=1.0, posinf=1.0, neginf=0.0,
        )
        labels         = np.asarray(labels, dtype=int)
        N, num_classes = scores.shape
        refined        = labels.astype(float).copy()

        target_list = getattr(args, "targert_list", [])
        alpha_tag   = getattr(args, "alpha", "")
        save_dir    = f"gmm_debug_plots{alpha_tag}"
        epsilon     = getattr(args, "epsilon", 0.05)
        fitter      = GMMFitter(n_components=self.n_components)

        for c in range(num_classes):
            col_scores = scores[:, c]
            col_labels = labels[:, c]
            stats      = {"pos_origin": 0, "neg_origin": 0,
                          "pos_flipped_to_neg": 0, "neg_flipped_to_pos": 0,
                          "soft_labels": 0}

            for target_y in (1, 0):
                mask          = col_labels == target_y
                subset_scores = col_scores[mask]
                stats_key     = "pos_origin" if target_y == 1 else "neg_origin"
                stats[stats_key] = int(mask.sum())

                if len(subset_scores) <= self.n_components:
                    continue

                prob_dict = fitter.fit_1d(subset_scores)
                if prob_dict is None:
                    continue

                # 視覺化（僅指定目標 label）
                if c in target_list:
                    visualize_gmm(
                        subset_scores,
                        class_name=c,
                        subset_type=f"{'Pos' if target_y == 1 else 'Neg'}"
                                    f"_Original{alpha_tag}",
                        save_dir=save_dir,
                        n_components=self.n_components,
                        label_index_path=getattr(
                            args, "label_index_path",
                            "dataset/AAPD/label_to_index.json",
                        ),
                        encoder_name=getattr(args, "encoder_name", ""),
                    )

                # apply_epsilon_band 回傳「clean 程度」（1=clean, 0=noisy）
                clean_degree = apply_epsilon_band(
                    prob_dict, self.n_components, epsilon=epsilon,
                )

                # target_y=1 → clean_degree 直接當 soft label
                # target_y=0 → 翻轉（clean → 維持 0，noisy → 變 1）
                new_vals = clean_degree if target_y == 1 else 1.0 - clean_degree
                refined[mask, c] = new_vals

                if target_y == 1:
                    stats["pos_flipped_to_neg"] = int((new_vals == 0.0).sum())
                else:
                    stats["neg_flipped_to_pos"] = int((new_vals == 1.0).sum())
                stats["soft_labels"] += int(
                    ((new_vals > 0.0) & (new_vals < 1.0)).sum()
                )

            self._log_perlabel_stats(c, stats)

        logger.info(f"[GMMNoiseFilter] 完成 {num_classes} 個類別的獨立校正。")
        return refined

    @staticmethod
    def _log_perlabel_stats(c: int, stats: dict) -> None:
        logger.info(
            f"[GMMNoiseFilter] label={c} "
            f"pos_origin={stats['pos_origin']} neg_origin={stats['neg_origin']} "
            f"pos→neg={stats['pos_flipped_to_neg']} "
            f"neg→pos={stats['neg_flipped_to_pos']} "
            f"soft={stats['soft_labels']}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2. IntersectionGMM3Filter
# ─────────────────────────────────────────────────────────────────────────────

class IntersectionGMM3Filter:

    """
    Per-label GMM(n) suspicious mask，取 CD ∩ Gap 的 AND。

    對每個 label c:
    1. 在 label=1 的子集上fit GMM
    2. 取 mean 最大的 component 作為 suspicious cluster
    3. cd 與 gap 兩個 mask 取 AND

    Parameters
    ----------
    n_components : int  GMM component 數，預設 3
    """

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components

    def intersect(
        self,
        cd_scores:  np.ndarray,
        gap_scores: np.ndarray,
        labels:     np.ndarray,
    ) -> np.ndarray:
        """回傳 [N, C] bool mask（True = CD 與 Gap 都判定 suspicious）。"""
        labels = np.asarray(labels, dtype=int)
        N, C   = labels.shape
        cd_mask  = np.zeros((N, C), dtype=bool)
        gap_mask = np.zeros((N, C), dtype=bool)

        for c in range(C):
            cd_mask[:, c]  = self._gmm_max_mean_mask(cd_scores[:, c],  labels[:, c])
            gap_mask[:, c] = self._gmm_max_mean_mask(gap_scores[:, c], labels[:, c])

        inter = cd_mask & gap_mask
        logger.info(
            f"[IntersectionGMM3Filter] "
            f"cd={cd_mask.sum()} gap={gap_mask.sum()} intersect={inter.sum()}"
        )
        return inter

    def _gmm_max_mean_mask(self, score: np.ndarray, label: np.ndarray) -> np.ndarray:
        """單一 label：fit GMM → 取 mean 最大的 component → 回傳 [N] bool mask。"""
        N    = len(score)
        mask = np.zeros(N, dtype=bool)
        pos  = np.where(label == 1)[0]
        if len(pos) < self.n_components * 2:
            return mask

        x = score[pos].reshape(-1, 1)
        try:
            gm = GMM(
                n_components=self.n_components,
                max_iter=200, tol=1e-3, reg_covar=1e-4, random_state=0,
            ).fit(x)
        except Exception as e:
            logger.warning(f"[IntersectionGMM3Filter] fit 失敗：{e}，跳過。")
            return mask

        susp = int(np.argmax(np.ravel(gm.means_)))
        mask[pos[gm.predict(x) == susp]] = True
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. StageOneRELFlipper
# ─────────────────────────────────────────────────────────────────────────────

class StageOneRELFlipper:

    """
    Stage 1 FP 翻轉器（TwoStagePipeline 的 helper）。

    對每個 label c，取 noisy_labels[:, c] == 1 的樣本中
    REL 分數最高的 top_ratio，將該 cell 翻轉為 0。

    Parameters
    ----------
    top_ratio : float  翻轉比例，預設 0.01（1%）
    """

    def __init__(self, top_ratio: float = 0.01) -> None:
        self.top_ratio = top_ratio

    def apply(
        self,
        rel_scores:    np.ndarray,
        noisy_labels:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        rel_scores   : [N, C] 標準化 REL 分數
        noisy_labels : [N, C] 原始 noisy 標籤（0/1）

        Returns
        -------
        flipped   : [N, C] float32  翻轉後的標籤
        flip_mask : [N, C] bool     被翻轉的 cell 位置
        """
        rel       = np.asarray(rel_scores,   dtype=float)
        labels    = np.asarray(noisy_labels, dtype=int)
        N, C      = labels.shape
        flipped   = labels.astype(np.float32).copy()
        flip_mask = np.zeros((N, C), dtype=bool)

        total = 0
        for c in range(C):
            pos_idx = np.where(labels[:, c] == 1)[0]
            if len(pos_idx) < 4:
                continue
            top_k      = max(1, int(len(pos_idx) * self.top_ratio))
            local_top  = np.argsort(rel[pos_idx, c])[-top_k:]
            cand       = pos_idx[local_top]
            flipped[cand, c]   = 0.0
            flip_mask[cand, c] = True
            total += len(cand)

        logger.info(
            f"[StageOneRELFlipper] top_ratio={self.top_ratio} "
            f"flipped={total} cells "
            f"({total / max(int(labels.sum()), 1):.2%} of positive cells)"
        )
        return flipped, flip_mask