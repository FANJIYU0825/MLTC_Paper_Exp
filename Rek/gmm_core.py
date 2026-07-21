"""
gmm_core.py
===========
GMM 演算法核心：fit、後驗機率排序、soft label 轉換。

此模組不依賴任何專案內部模組，可獨立使用。

公開 API
--------
GMMFitter                  ← fit 1D / 2D GMM，回傳統一格式的 prob_dict
apply_epsilon_band()       ← 將 prob_dict 轉換為 soft label（支援 2~5 components）

prob_dict 格式
--------------
{
    'clean_probs' : np.ndarray [N],   # component 0（最乾淨）的後驗機率
    'noisy_probs' : np.ndarray [N],   # component -1（最可疑）的後驗機率
    'mid_probs'   : np.ndarray [N],   # 僅 3-comp 存在
    'mid_probs_0' : np.ndarray [N],   # 4-comp / 5-comp 的中間 component
    'mid_probs_1' : np.ndarray [N],   # 5-comp 才有
    ...
}

排序規則
--------
1D : 按 component mean 升序  → idx 0 = 低分群 (clean)
2D : 按 component mean norm 升序 → idx 0 = 原點附近 (clean)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture as GMM


# ─────────────────────────────────────────────────────────────────────────────
# 型別別名
# ─────────────────────────────────────────────────────────────────────────────

ProbDict   = dict[str, np.ndarray]
ClipRange  = Optional[Tuple[float, float]]


# ─────────────────────────────────────────────────────────────────────────────
# 1. GMMFitter
# ─────────────────────────────────────────────────────────────────────────────

class GMMFitter:
    """
    單一職責：fit GMM 並回傳排序後的後驗機率字典（prob_dict）。

    消除原始程式碼中散落在五個類別裡的重複 GMM 邏輯：
        - GMMNoiseFilter._run_gmm_on_subset
        - LabelRefiner._run_gmm
        - TwoStageRELFPCorrector._run_1d_gmm
        - TwoStageRELFPCorrector._run_2d_gmm
        - IntersectionGMM3Filter._suspicious_mask_one

    Parameters
    ----------
    n_components : int
        GMM 的 component 數量，支援 2~5。
    max_iter : int
        EM 演算法最大迭代次數。
    reg_covar : float
        協方差矩陣的正則化項，防止數值奇異。
    random_state : int
        隨機種子，確保結果可重現。
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter:     int = 200,
        reg_covar:    float = 1e-4,
        random_state: int = 0,
    ) -> None:
        if n_components < 2:
            raise ValueError(f"n_components 必須 >= 2，目前為 {n_components}。")
        self.n_components = n_components
        self.max_iter     = max_iter
        self.reg_covar    = reg_covar
        self.random_state = random_state

    # ── 公開方法 ─────────────────────────────────────────────────────────────

    def fit_1d(self, scores: np.ndarray) -> ProbDict | None:
        """
        在 1D 分數上 fit GMM。
        排序規則：component mean 升序，idx 0 = 低分（clean）。

        Parameters
        ----------
        scores : np.ndarray, shape [N]

        Returns
        -------
        ProbDict 或 None（樣本數不足以 fit 時）
        """
        scores = np.asarray(scores, dtype=float).ravel()
        if len(scores) <= self.n_components:
            return None

        gmm = self._make_gmm()
        try:
            gmm.fit(scores.reshape(-1, 1))
        except Exception as e:
            raise RuntimeError(f"[GMMFitter.fit_1d] GMM fit 失敗：{e}") from e

        sorted_idx = np.argsort(gmm.means_.flatten())          # mean 升序
        return self._build_prob_dict(gmm, scores.reshape(-1, 1), sorted_idx)

    def fit_2d(self, features: np.ndarray) -> ProbDict | None:
        """
        在 2D 特徵上 fit GMM。
        排序規則：component mean 的 L2 norm 升序，idx 0 = 原點附近（clean）。

        Parameters
        ----------
        features : np.ndarray, shape [N, 2]
            通常為 (−CD, Gap) 或 (CD, Gap)，方向由呼叫端決定。

        Returns
        -------
        ProbDict 或 None（樣本數不足以 fit 時）
        """
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != 2:
            raise ValueError(
                f"features 必須是 [N, 2]，目前為 {features.shape}。"
            )
        if len(features) <= self.n_components:
            return None

        gmm = self._make_gmm()
        try:
            gmm.fit(features)
        except Exception as e:
            raise RuntimeError(f"[GMMFitter.fit_2d] GMM fit 失敗：{e}") from e

        sorted_idx = np.argsort(np.linalg.norm(gmm.means_, axis=1))  # norm 升序
        return self._build_prob_dict(gmm, features, sorted_idx)

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _make_gmm(self) -> GMM:
        return GMM(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=1e-3,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        )

    def _build_prob_dict(
        self,
        gmm: GMM,
        X: np.ndarray,
        sorted_idx: np.ndarray,
    ) -> ProbDict:
        """
        將 GMM 後驗機率依排序後的 component index 組成 prob_dict。

        sorted_idx[0]  → 'clean_probs'
        sorted_idx[-1] → 'noisy_probs'
        sorted_idx[1:-1] →
            3-comp : 'mid_probs'
            4-comp : 'mid_probs_0', 'mid_probs_1'
            5-comp : 'mid_probs_0', 'mid_probs_1', 'mid_probs_2'
        """
        probs_all = gmm.predict_proba(X)

        result: ProbDict = {
            "clean_probs": probs_all[:, sorted_idx[0]],
            "noisy_probs": probs_all[:, sorted_idx[-1]],
        }

        mid_indices = sorted_idx[1:-1]
        if len(mid_indices) == 1:
            # 3-comp：保持向後相容，使用不帶數字的 key
            result["mid_probs"] = probs_all[:, mid_indices[0]]
        else:
            for i, idx in enumerate(mid_indices):
                result[f"mid_probs_{i}"] = probs_all[:, idx]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. apply_epsilon_band
# ─────────────────────────────────────────────────────────────────────────────

# 各 component 數對應的 soft label gradient（clean → noisy 方向）
_GRADIENTS: dict[int, list[float]] = {
    2: [],                          # 2-comp 使用 epsilon-band，不用 gradient
    3: [1.0, 0.5, 0.0],
    4: [1.0, 0.7, 0.3, 0.0],
    5: [1.0, 0.75, 0.5, 0.25, 0.0],
}


def apply_epsilon_band(
    prob_dict:    ProbDict,
    n_components: int,
    epsilon:      float = 0.05,
) -> np.ndarray:
    """
    將 GMMFitter 回傳的 prob_dict 轉換為 soft label。

    回傳值語意：1.0 = 確定乾淨（clean）、0.0 = 確定雜訊（noisy）、
    中間值 = 不確定（soft label）。

    規則
    ----
    n_components=2 : epsilon-band
        prob_clean > 0.5 + ε  →  1.0
        prob_clean < 0.5 - ε  →  0.0
        中間帶                →  prob_clean（soft）

    n_components=3 : argmax → gradient [1.0, 0.5, 0.0]
    n_components=4 : argmax → gradient [1.0, 0.7, 0.3, 0.0]
    n_components=5 : argmax → gradient [1.0, 0.75, 0.5, 0.25, 0.0]

    Parameters
    ----------
    prob_dict : ProbDict
        由 GMMFitter.fit_1d / fit_2d 回傳的後驗機率字典。
    n_components : int
        GMM 的 component 數，必須與 prob_dict 一致。
    epsilon : float
        epsilon-band 的半寬，僅 n_components=2 時使用。

    Returns
    -------
    np.ndarray [N], 值域 {0.0, soft, 1.0}

    Raises
    ------
    ValueError
        若 n_components 不在支援範圍（2~5）。
    """
    if n_components not in _GRADIENTS:
        raise ValueError(
            f"n_components={n_components} 不支援，請使用 2~5。"
        )

    # ── 2-comp：epsilon-band ─────────────────────────────────────────────────
    if n_components == 2:
        p = np.nan_to_num(prob_dict["clean_probs"], nan=0.5)
        return np.where(
            p > 0.5 + epsilon, 1.0,
            np.where(p < 0.5 - epsilon, 0.0, p),
        )

    # ── 3~5-comp：argmax + gradient ─────────────────────────────────────────
    gradient = _GRADIENTS[n_components]

    # 依序取出 clean / mid_* / noisy 的機率，組成 [n_comp, N] 矩陣
    keys = _get_ordered_keys(n_components)
    stack = np.vstack([prob_dict[k] for k in keys])   # [n_comp, N]

    winners = np.argmax(stack, axis=0)                 # [N]
    out = np.zeros(stack.shape[1], dtype=float)
    for i, val in enumerate(gradient):
        out[winners == i] = val
    return out


def _get_ordered_keys(n_components: int) -> list[str]:
    """
    回傳 prob_dict 中 clean → mid_* → noisy 的 key 順序。
    供 apply_epsilon_band 內部使用。
    """
    if n_components == 3:
        return ["clean_probs", "mid_probs", "noisy_probs"]
    mid_keys = [f"mid_probs_{i}" for i in range(n_components - 2)]
    return ["clean_probs"] + mid_keys + ["noisy_probs"]