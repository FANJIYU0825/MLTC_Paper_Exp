"""
correctors.py
=============
標籤校正策略（Corrector / Refiner 語意）。

與 filters.py 的區別：
    filters.py   → 判斷「哪些 cell 是雜訊」，回傳 mask
    correctors.py → 決定「把雜訊 cell 改成多少」，回傳校正後的 label 矩陣

公開 API
--------
LabelRefiner                  ← per-label GMM，正負樣本分開校正
RELOnlyCorrector              ← Stage 1 only：REL top-k 直接翻轉（基線）
TwoStageRELFPCorrector        ← Stage 1 REL top-k + Stage 2 2D/1D GMM
FrequencyAware1DGMMCorrector  ← 頻率感知 1D GMM，依 CD-Gap 相關性選融合策略

依賴
----
gmm_core.py — GMMFitter, apply_epsilon_band
"""

from __future__ import annotations

import numpy as np

from .gmm_core import GMMFitter, apply_epsilon_band

try:
    from util.logger import logger
    from util.experence import (
        visualize_gmm,
        visualize_2d_gmm_candidates,
        visualize_gmm_cluster_purity_k,
        visualize_1d_gmm_features,
    )
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

    # No-op stubs，避免視覺化失敗中斷主流程
    def visualize_gmm(*a, **kw): pass
    def visualize_2d_gmm_candidates(*a, **kw): pass
    def visualize_gmm_cluster_purity_k(*a, **kw): pass
    def visualize_1d_gmm_features(*a, **kw): pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. LabelRefiner
# ─────────────────────────────────────────────────────────────────────────────

class LabelRefiner:
    """
    Per-label GMM 校正器。

    對每個 label c，分別對正樣本（lab=1）與負樣本（lab=0）
    執行 n_components-GMM，再以 apply_epsilon_band 轉換成 soft label。

    Parameters
    ----------
    n_components : int   GMM component 數，支援 2~5，預設 2
    random_state : int   隨機種子
    """

    def __init__(self, n_components: int = 2, random_state: int = 0) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def refine(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        args,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        scores : [N, C] float  HSM 分數（已標準化）
        labels : [N, C] int    原始 noisy 標籤
        args   : 含 args.epsilon, args.targert_list, args.alpha,
                 args.label_index_path, args.encoder_name

        Returns
        -------
        refined_labels : [N, C] float
        """
        logger.info(f"[LabelRefiner] n_components={self.n_components}")

        scores         = np.nan_to_num(
            np.asarray(scores, dtype=float), nan=1.0, posinf=1.0, neginf=0.0
        )
        labels         = np.asarray(labels, dtype=int)
        refined_labels = labels.astype(float).copy()
        N, C           = scores.shape

        target_list = getattr(args, "targert_list", [])
        save_dir    = f"gmm_debug_plots{getattr(args, 'alpha', '')}"
        fitter      = GMMFitter(
            n_components=self.n_components, random_state=self.random_state
        )

        for c in range(C):
            col_scores = scores[:, c]
            col_labels = labels[:, c]
            stats      = {"1->0": 0, "0->1": 0, "soft": 0,
                          "pos_orig": 0, "neg_orig": 0}

            for target_y in (1, 0):
                mask           = col_labels == target_y
                subset_scores  = col_scores[mask]
                key            = "pos_orig" if target_y == 1 else "neg_orig"
                stats[key]     = int(mask.sum())

                if len(subset_scores) <= self.n_components:
                    continue

                prob_dict = fitter.fit_1d(subset_scores)
                if prob_dict is None:
                    continue

                if c in target_list:
                    visualize_gmm(
                        subset_scores,
                        class_name=c,
                        subset_type=f"{'Pos' if target_y == 1 else 'Neg'}"
                                    f"_Original{getattr(args, 'alpha', '')}",
                        save_dir=save_dir,
                        n_components=self.n_components,
                        label_index_path=getattr(
                            args, "label_index_path",
                            "dataset/AAPD/label_to_index.json"
                        ),
                    )

                # apply_epsilon_band 回傳「clean 程度」（1=clean, 0=noisy）
                clean_degree = apply_epsilon_band(
                    prob_dict, self.n_components,
                    epsilon=getattr(args, "epsilon", 0.05),
                )

                # 轉換為實際標籤：
                #   target_y=1 → clean_degree 直接當 soft label
                #   target_y=0 → 翻轉（clean → 保留 0，noisy → 變 1）
                new_vals = clean_degree if target_y == 1 else 1.0 - clean_degree
                refined_labels[mask, c] = new_vals

                if target_y == 1:
                    stats["1->0"] = int((new_vals == 0.0).sum())
                else:
                    stats["0->1"] = int((new_vals == 1.0).sum())
                stats["soft"] += int(
                    ((new_vals > 0.0) & (new_vals < 1.0)).sum()
                )

            logger.info(
                f"  Label {c} | P/N: {stats['pos_orig']}/{stats['neg_orig']} | "
                f"Flipped 1->0:{stats['1->0']} 0->1:{stats['0->1']} | "
                f"Soft:{stats['soft']}"
            )

        return refined_labels


# ─────────────────────────────────────────────────────────────────────────────
# 2. RELOnlyCorrector
# ─────────────────────────────────────────────────────────────────────────────

class RELOnlyCorrector:
    """
    Stage 1 only 基線：REL top-k 直接翻轉。

    對每個 label c，取 label=1 樣本中 REL 最高的 top_ratio，翻轉為 0。
    作為 TwoStageRELFPCorrector 的對照組。

    Parameters
    ----------
    top_ratio : float  翻轉比例，預設 0.01
    """

    def __init__(self, top_ratio: float = 0.01) -> None:
        self.top_ratio = top_ratio

    def correct(
        self,
        rel_scores: np.ndarray,
        labels:     np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns
        -------
        corrected : [N, C] float
        """
        rel       = np.asarray(rel_scores, dtype=float)
        labels    = np.asarray(labels,     dtype=int)
        corrected = labels.astype(float).copy()
        total     = 0

        for c in range(labels.shape[1]):
            pos_idx = np.where(labels[:, c] == 1)[0]
            if len(pos_idx) < 4:
                continue
            top_k       = max(1, int(len(pos_idx) * self.top_ratio))
            local_top   = np.argsort(rel[pos_idx, c])[-top_k:]
            cand        = pos_idx[local_top]
            corrected[cand, c] = 0.0
            total += len(cand)
            logger.info(
                f"  [RELOnly c={c}] flipped={len(cand)} "
                f"({len(cand)/len(pos_idx):.1%} of positives)"
            )

        logger.info(f"[RELOnlyCorrector] total_flipped={total}")
        return corrected


# ─────────────────────────────────────────────────────────────────────────────
# 3. TwoStageRELFPCorrector
# ─────────────────────────────────────────────────────────────────────────────

class TwoStageRELFPCorrector:
    """
    兩階段 FP 校正器（僅針對 label=1 的樣本）。

    Stage 1 — REL 篩選（per-label top_ratio 直接翻轉）：
        REL 最高的 top_ratio 樣本視為確定 FP，直接翻轉為 0。

    Stage 2 — 剩餘樣本以 GMM 做 soft label 校正：
        feature_mode='2d'  → (CD, Gap) 2D GMM
        feature_mode='cd'  → −CD 1D GMM
        feature_mode='gap' → Gap 1D GMM

    Parameters
    ----------
    top_ratio     : float   Stage 1 直接翻轉比例，預設 0.05
    epsilon       : float   epsilon-band 半寬，預設 0.05
    n_components  : int     GMM component 數，預設 2
    feature_mode  : str     '2d' | 'cd' | 'gap'
    """

    def __init__(
        self,
        top_ratio:    float = 0.05,
        epsilon:      float = 0.05,
        n_components: int   = 2,
        feature_mode: str   = "2d",
    ) -> None:
        if feature_mode not in ("2d", "cd", "gap"):
            raise ValueError(
                f"feature_mode='{feature_mode}' 不合法，請選 '2d'、'cd' 或 'gap'。"
            )
        self.top_ratio    = top_ratio
        self.epsilon      = epsilon
        self.n_components = n_components
        self.feature_mode = feature_mode

    # ── 公開方法 ─────────────────────────────────────────────────────────────

    def correct(
        self,
        rel_scores:  np.ndarray,
        cd_scores:   np.ndarray,
        gap_scores:  np.ndarray,
        labels:      np.ndarray,
        args,
        y_true:      np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rel_scores  : [N, C] 標準化 REL 分數
        cd_scores   : [N, C] 標準化 CD 分數（masked，lab=1 → [-1, 0]）
        gap_scores  : [N, C] 標準化 Gap 分數（masked，lab=1 → 正值）
        labels      : [N, C] 原始 noisy 標籤
        args        : 含 args.epsilon
        y_true      : [N, C] 真實標籤（可選，用於日誌）

        Returns
        -------
        corrected : [N, C] float
        """
        rel       = np.asarray(rel_scores, dtype=float)
        cd        = np.asarray(cd_scores,  dtype=float)
        gap       = np.asarray(gap_scores, dtype=float)
        labels    = np.asarray(labels,     dtype=int)
        N, C      = labels.shape
        corrected = labels.astype(float).copy()
        epsilon   = getattr(args, "epsilon", self.epsilon)

        total_candidates = total_flipped = total_soft = 0

        for c in range(C):
            pos_idx = np.where(labels[:, c] == 1)[0]
            if len(pos_idx) < 4:
                continue

            # ── Stage 1 ──────────────────────────────────────────────────────
            flipped_idx, rest_idx = self._stage1_split(rel[:, c], pos_idx)
            corrected[flipped_idx, c] = 0.0
            total_flipped += len(flipped_idx)
            logger.info(f"  [TwoStage c={c}] Stage1 flipped={len(flipped_idx)}")

            # ── Stage 2 ──────────────────────────────────────────────────────
            if len(rest_idx) < 4:
                continue
            total_candidates += len(rest_idx)

            cd_c  = cd[rest_idx, c]
            gap_c = gap[rest_idx, c]

            if y_true is not None:
                self._log_pool_purity(c, y_true[rest_idx, c])

            new_labels = self._stage2_gmm(cd_c, gap_c, epsilon, args, c, y_true, rest_idx)
            if new_labels is None:
                continue

            corrected[rest_idx, c] = new_labels
            gmm_flipped             = int((new_labels == 0.0).sum())
            soft                    = int(
                ((new_labels > 0.0) & (new_labels < 1.0)).sum()
            )
            total_flipped += gmm_flipped
            total_soft    += soft
            logger.info(
                f"  [TwoStage c={c}] Stage2 candidates={len(rest_idx)}, "
                f"flipped={gmm_flipped}, soft={soft}"
            )

        logger.info(
            f"[TwoStageRELFPCorrector] "
            f"stage2_candidates={total_candidates} "
            f"flipped={total_flipped} soft={total_soft}"
        )
        return corrected

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _stage1_split(
        self,
        rel_c:      np.ndarray,
        pos_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """回傳 (flipped_global_idx, rest_global_idx)。

        rel_c 為整欄 [N] 分數；先取 pos_indices 子集後再排序，
        避免 argsort 出來的 local index 超出 pos_indices 範圍。
        """
        top_k      = max(1, int(len(pos_indices) * self.top_ratio))
        rel_pos    = rel_c[pos_indices]               # [len(pos_indices)]
        order      = np.argsort(rel_pos)
        local_top  = order[-top_k:]
        local_rest = order[:-top_k]
        return pos_indices[local_top], pos_indices[local_rest]

    def _stage2_gmm(
        self,
        cd_c:    np.ndarray,
        gap_c:   np.ndarray,
        epsilon: float,
        args,
        c:       int,
        y_true:  np.ndarray | None,
        rest_idx: np.ndarray,
    ) -> np.ndarray | None:
        """Stage 2 GMM，回傳 soft label 或 None（fit 失敗時）。"""
        target_list     = getattr(args, "targert_list", [])
        save_dir        = getattr(args, "Resutl_dir", "gmm_debug_plots") + "/twostage_plots"
        label_idx_path  = getattr(args, "label_index_path", None)
        enc_name        = getattr(args, "encoder_name", "")
        true_c          = y_true[rest_idx, c] if y_true is not None else None
        fitter          = GMMFitter(n_components=self.n_components)

        try:
            if self.feature_mode == "2d":
                features  = np.column_stack([cd_c, gap_c])
                prob_dict = fitter.fit_2d(features)
                if prob_dict is None:
                    return None
                if c in target_list:
                    visualize_2d_gmm_candidates(
                        cd_vals=cd_c, gap_vals=gap_c,
                        prob_clean=prob_dict["clean_probs"],
                        label_index=c, save_dir=save_dir,
                        label_index_path=label_idx_path,
                        encoder_name=enc_name, true_labels=true_c,
                    )
                if c in target_list and true_c is not None:
                    visualize_gmm_cluster_purity_k(
                        cd_vals=cd_c, gap_vals=gap_c, true_labels=true_c,
                        label_index=c, k_values=(2, 3, 4, 5),
                        save_dir=save_dir,
                        label_index_path=label_idx_path,
                        encoder_name=enc_name,
                    )
            else:
                scores_1d = -cd_c if self.feature_mode == "cd" else gap_c
                prob_dict = fitter.fit_1d(scores_1d)
                if prob_dict is None:
                    return None
                if c in target_list and true_c is not None:
                    visualize_1d_gmm_features(
                        cd_vals=cd_c, gap_vals=gap_c, true_labels=true_c,
                        label_index=c, n_components=self.n_components,
                        save_dir=save_dir,
                        label_index_path=label_idx_path,
                        encoder_name=f"{enc_name}_{self.feature_mode}",
                    )

        except Exception as e:
            logger.warning(
                f"  [TwoStage c={c}] Stage2 GMM "
                f"(n={self.n_components}, mode={self.feature_mode}) 失敗：{e}，跳過。"
            )
            return None

        return apply_epsilon_band(prob_dict, self.n_components, epsilon)

    @staticmethod
    def _log_pool_purity(c: int, true_c: np.ndarray) -> None:
        n_fp = int((true_c == 0).sum())
        n_tp = int((true_c == 1).sum())
        ratio = n_fp / len(true_c) if len(true_c) > 0 else 0.0
        logger.info(
            f"  [TwoStage c={c}] Stage2 pool FP率: "
            f"{n_fp}/{len(true_c)} ({ratio:.1%})  TP={n_tp} FP={n_fp}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. FrequencyAware1DGMMCorrector
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyAware1DGMMCorrector:
    """
    頻率感知 1D GMM 校正器（FP 導向）。

    依標籤正樣本數量分組（head / middle / tail），
    再根據 CD-Gap Pearson 相關係數選擇融合策略（max / mean / gap），
    最後以 noisy prob 方向的 epsilon-band 校正正樣本標籤。

    策略選擇規則
    ------------
    pos_count < MIN_CD_SAMPLES 或 tail → 'gap'（CD 原型估計不穩）
    corr > CORR_HIGH (0.6)             → 'mean'（兩信號一致）
    corr < CORR_LOW  (0.3)             → 'max'（兩信號不一致）
    其餘依頻率組                        → head='max', middle='mean'

    Parameters
    ----------
    n_components  : int    GMM component 數，預設 2
    head_epsilon  : float  head 標籤的 epsilon，預設 0.03
    mid_epsilon   : float  middle 標籤的 epsilon，預設 0.05
    tail_epsilon  : float  tail 標籤的 epsilon，預設 0.10
    """

    HEAD_THRESH    = 4500
    MID_THRESH     = 2000
    CORR_HIGH      = 0.6
    CORR_LOW       = 0.3
    MIN_CD_SAMPLES = 100

    def __init__(
        self,
        n_components: int   = 2,
        head_epsilon: float = 0.03,
        mid_epsilon:  float = 0.05,
        tail_epsilon: float = 0.10,
    ) -> None:
        self.n_components = n_components
        self.head_epsilon = head_epsilon
        self.mid_epsilon  = mid_epsilon
        self.tail_epsilon = tail_epsilon

    # ── 公開方法 ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        cd_scores:  np.ndarray,
        gap_scores: np.ndarray,
        labels:     np.ndarray,
        save_path:  str | None = None,
    ):
        """
        輸出每個標籤的分布統計與自動建議的融合策略（DataFrame）。

        Returns
        -------
        pd.DataFrame  columns: label, freq_group, pos_count,
                               cd_mean, cd_std, gap_mean, gap_std,
                               cd_gap_corr, strategy
        """
        import pandas as pd
        freq_group, _ = self._classify_labels(labels)
        records       = []

        for c in range(labels.shape[1]):
            pos_idx = np.where(labels[:, c] == 1)[0]
            n       = len(pos_idx)
            if n < 4:
                records.append({
                    "label": c, "freq_group": freq_group[c], "pos_count": n,
                    "cd_mean": np.nan, "cd_std": np.nan,
                    "gap_mean": np.nan, "gap_std": np.nan,
                    "cd_gap_corr": np.nan, "strategy": "skip",
                })
                continue

            cd_c  = cd_scores[pos_idx, c]
            gap_c = gap_scores[pos_idx, c]
            corr  = self._pearson(cd_c, gap_c)
            records.append({
                "label":       c,
                "freq_group":  freq_group[c],
                "pos_count":   n,
                "cd_mean":     round(float(cd_c.mean()),  4),
                "cd_std":      round(float(cd_c.std()),   4),
                "gap_mean":    round(float(gap_c.mean()), 4),
                "gap_std":     round(float(gap_c.std()),  4),
                "cd_gap_corr": round(corr, 4),
                "strategy":    self._select_strategy(freq_group[c], corr, n),
            })

        df = pd.DataFrame(records)
        logger.info(
            "[FA1DGMM.analyze] strategy distribution:\n"
            + df.groupby("strategy")["label"].count().to_string()
        )
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"[FA1DGMM.analyze] saved → {save_path}")
        return df

    def compute_suspicion(
        self,
        cd_scores:  np.ndarray,
        gap_scores: np.ndarray,
        labels:     np.ndarray,
    ) -> np.ndarray:
        """
        回傳 [N, C] suspicion 矩陣，僅 label=1 位置有值。

        值越高代表「是 FP 的嫌疑越大」。
        """
        N, C       = labels.shape
        freq_group, _ = self._classify_labels(labels)
        suspicion  = np.zeros((N, C), dtype=float)
        fitter     = GMMFitter(n_components=self.n_components)

        for c in range(C):
            pos_idx = np.where(labels[:, c] == 1)[0]
            if len(pos_idx) < 4:
                continue

            cd_c  = cd_scores[pos_idx, c]
            gap_c = gap_scores[pos_idx, c]
            corr  = self._pearson(cd_c, gap_c)

            strategy = self._select_strategy(freq_group[c], corr, len(pos_idx))

            # 回傳「noisy（FP）後驗機率」
            def _noisy_prob(scores_1d):
                pd_ = fitter.fit_1d(scores_1d)
                if pd_ is None:
                    return np.zeros(len(scores_1d))
                return pd_["noisy_probs"]

            cd_noisy  = _noisy_prob(cd_c)
            gap_noisy = _noisy_prob(gap_c)

            if strategy == "max":
                combined = np.maximum(cd_noisy, gap_noisy)
            elif strategy == "mean":
                combined = (cd_noisy + gap_noisy) / 2.0
            else:   # 'gap'
                combined = gap_noisy

            logger.debug(
                f"  [FA1DGMM c={c:>2} | {freq_group[c]:6} "
                f"| corr={corr:+.3f}] strategy={strategy}"
            )
            suspicion[pos_idx, c] = combined

        return suspicion

    def correct(
        self,
        cd_scores:  np.ndarray,
        gap_scores: np.ndarray,
        labels:     np.ndarray,
    ) -> np.ndarray:
        """
        依資料驅動 epsilon-band 校正所有正樣本。

        Returns
        -------
        corrected : [N, C] float
        """
        freq_group, _ = self._classify_labels(labels)
        suspicion     = self.compute_suspicion(cd_scores, gap_scores, labels)
        corrected     = labels.astype(float).copy()
        eps_map       = {
            "head":   self.head_epsilon,
            "middle": self.mid_epsilon,
            "tail":   self.tail_epsilon,
        }

        for c in range(labels.shape[1]):
            pos_idx = np.where(labels[:, c] == 1)[0]
            if len(pos_idx) < 4:
                continue
            eps     = eps_map[freq_group[c]]
            # suspicion = noisy prob；轉換：high noisy → 0.0，low noisy → 1.0
            p       = np.nan_to_num(suspicion[pos_idx, c], nan=0.5)
            new_lab = np.where(
                p > 0.5 + eps, 0.0,
                np.where(p < 0.5 - eps, 1.0, 1.0 - p),
            )
            corrected[pos_idx, c] = new_lab
            logger.info(
                f"  [FA1DGMM c={c:>2} | {freq_group[c]:6}] "
                f"pos={len(pos_idx)} "
                f"flipped={(new_lab == 0.0).sum()} "
                f"soft={((new_lab > 0.0) & (new_lab < 1.0)).sum()} "
                f"eps={eps}"
            )

        return corrected

    def progressive_correct(
        self,
        cd_scores:  np.ndarray,
        gap_scores: np.ndarray,
        labels:     np.ndarray,
        y_true:     np.ndarray,
        n_rounds:   int = 10,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        按可疑分數降序逐步翻轉，每輪回報 FP Recall。

        Returns
        -------
        corrected   : [N, C] float
        round_stats : list[dict]
        """
        suspicion              = self.compute_suspicion(cd_scores, gap_scores, labels)
        pos_rows, pos_cols     = np.where(labels == 1)
        sorted_order           = np.argsort(-suspicion[pos_rows, pos_cols])
        s_rows, s_cols         = pos_rows[sorted_order], pos_cols[sorted_order]
        n_pos                  = len(s_rows)
        batch_sz               = max(1, n_pos // n_rounds)

        corrected   = labels.astype(float).copy()
        fp_mask     = (y_true == 0) & (labels == 1)
        total_fp    = int(fp_mask.sum())
        round_stats = []

        for r in range(n_rounds):
            start = r * batch_sz
            end   = (r + 1) * batch_sz if r < n_rounds - 1 else n_pos
            corrected[s_rows[start:end], s_cols[start:end]] = 0.0

            bin_pred     = (corrected >= 0.5).astype(int)
            fix_fp       = int(((bin_pred == 0) & fp_mask).sum())
            fp_recall    = fix_fp / total_fp if total_fp > 0 else 0.0

            flipped_mask = np.zeros_like(corrected, dtype=bool)
            flipped_mask[s_rows[:end], s_cols[:end]] = True
            fp_precision = int((flipped_mask & fp_mask).sum()) / end if end > 0 else 0.0

            stat = {
                "round":              r + 1,
                "cumulative_flipped": end,
                "flip_ratio":         end / n_pos,
                "fix_fp":             fix_fp,
                "miss_fp":            total_fp - fix_fp,
                "total_fp":           total_fp,
                "fp_recall":          fp_recall,
                "fp_precision":       fp_precision,
            }
            round_stats.append(stat)
            logger.info(
                f"  [Progressive {r+1:>2}/{n_rounds}] "
                f"flipped={end}/{n_pos} ({end/n_pos:.1%})  "
                f"Fix_FP={fix_fp}  "
                f"FP_Recall={fp_recall:.3f}  FP_Prec={fp_precision:.3f}"
            )

        return corrected, round_stats

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _classify_labels(
        self,
        labels: np.ndarray,
    ) -> tuple[dict[int, str], np.ndarray]:
        pos_counts = labels.sum(axis=0).astype(int)
        freq_group = {}
        for c, cnt in enumerate(pos_counts):
            if cnt >= self.HEAD_THRESH:
                freq_group[c] = "head"
            elif cnt >= self.MID_THRESH:
                freq_group[c] = "middle"
            else:
                freq_group[c] = "tail"
        return freq_group, pos_counts

    def _select_strategy(
        self, group: str, corr: float, pos_count: int
    ) -> str:
        if pos_count < self.MIN_CD_SAMPLES or group == "tail":
            return "gap"
        if corr > self.CORR_HIGH:
            return "mean"
        if corr < self.CORR_LOW:
            return "max"
        return "max" if group == "head" else "mean"

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        corr = float(np.corrcoef(a, b)[0, 1])
        return corr if np.isfinite(corr) else 0.0