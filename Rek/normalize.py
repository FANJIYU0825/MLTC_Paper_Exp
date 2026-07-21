"""
normalize.py
============
純數學工具：所有標準化 / 前處理函式。

此模組不依賴任何專案內部模組，可獨立使用。

公開 API
--------
sanitize_array(x, fill_for_nonfinite)
minmax_normalize(matrix)
zscore_normalize(matrix, clip_range)
robust_zscore_normalize(matrix, clip_range)
robust_normalize(score_array, clip_percentile)   # 1D 用
normalize(arr, method, **kwargs)                  # 統一入口
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 型別別名
# ─────────────────────────────────────────────────────────────────────────────

NormMethod = Literal["minmax", "zscore", "robust_zscore"]
ClipRange  = Optional[Tuple[float, float]]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 前處理工具
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_array(
    x: np.ndarray,
    fill_for_nonfinite: float = 0.0,
) -> np.ndarray:
    """
    將陣列中所有非有限值（NaN / ±inf）替換為指定值。

    Parameters
    ----------
    x : array-like
    fill_for_nonfinite : float
        用來填補非有限值的替代數值，預設為 0.0。
        若要標記「無資料」語意，可傳入 -np.inf。

    Returns
    -------
    np.ndarray  (float64，複製品，不修改原始輸入)
    """
    x = np.asarray(x)
    if x.size == 0:
        return x
    out  = x.astype(float, copy=True)
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = fill_for_nonfinite
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Column-wise 標準化（適用 [N, C] 或 [N] 矩陣）
# ─────────────────────────────────────────────────────────────────────────────

def minmax_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Min-Max 標準化（column-wise）。

    將每個 column 縮放到 [0, 1]。
    若某 column 的 range < 1e-8（所有值相同），該 column 輸出全為 0。

    Parameters
    ----------
    matrix : np.ndarray, shape [N, C] or [N]

    Returns
    -------
    np.ndarray, 與輸入同形狀，值域 [0, 1]
    """
    out = matrix.astype(float, copy=True)

    squeezed = out.ndim == 1
    if squeezed:
        out = out.reshape(-1, 1)

    min_vals   = out.min(axis=0)
    max_vals   = out.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-8] = 1e-8   # 避免除以 0，輸出結果為 0

    out = (out - min_vals) / range_vals

    if squeezed:
        out = out.squeeze(1)
    return out


def zscore_normalize(
    matrix: np.ndarray,
    clip_range: ClipRange = None,
) -> np.ndarray:
    """
    Z-Score 標準化（column-wise）。

    每個 column 轉換為 (x - mean) / std，使 mean ≈ 0、std ≈ 1。
    若某 column std < 1e-8，視為常數，不進行縮放（std 置為 1.0）。

    Parameters
    ----------
    matrix : np.ndarray, shape [N, C] or [N]
    clip_range : (min, max) or None
        限制輸出 z-score 的範圍，例如 (-5, 5)。

    Returns
    -------
    np.ndarray, 與輸入同形狀
    """
    out = matrix.astype(float, copy=True)

    squeezed = out.ndim == 1
    if squeezed:
        out = out.reshape(-1, 1)

    mean_vals = out.mean(axis=0)
    std_vals  = out.std(axis=0)
    std_vals[std_vals < 1e-8] = 1.0   # 常數 column：std → 1，輸出全為 0

    out = (out - mean_vals) / std_vals

    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])

    if squeezed:
        out = out.squeeze(1)
    return out


def robust_zscore_normalize(
    matrix: np.ndarray,
    clip_range: ClipRange = None,
) -> np.ndarray:
    """
    Robust Z-Score 標準化（column-wise，使用中位數與 MAD）。

    對極端值更穩健，公式：(x - median) / (1.4826 × MAD)
    其中 1.4826 為常態分佈下將 MAD 轉換為標準差的係數。

    若某 column scale < 1e-8，視為常數，不進行縮放。

    Parameters
    ----------
    matrix : np.ndarray, shape [N, C] or [N]
    clip_range : (min, max) or None

    Returns
    -------
    np.ndarray, 與輸入同形狀
    """
    out = matrix.astype(float, copy=True)

    squeezed = out.ndim == 1
    if squeezed:
        out = out.reshape(-1, 1)

    median_vals = np.median(out, axis=0)
    mad_vals    = np.median(np.abs(out - median_vals), axis=0)
    scale       = 1.4826 * mad_vals
    scale[scale < 1e-8] = 1.0

    out = (out - median_vals) / scale

    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])

    if squeezed:
        out = out.squeeze(1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. 1D 專用（抗極端值）
# ─────────────────────────────────────────────────────────────────────────────

def robust_normalize(
    score_array: np.ndarray,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """
    1D 抗極端值歸一化。

    先將數值 clip 到第 clip_percentile 百分位數以下，
    再做 Min-Max 縮放到 [0, 1]。

    Parameters
    ----------
    score_array : array-like, 1D
    clip_percentile : float
        裁切上界的百分位數，預設 99.0（去除前 1% 的極端高值）。

    Returns
    -------
    np.ndarray, 值域 [0, 1]
    """
    arr   = np.asarray(score_array, dtype=float)
    limit = np.percentile(arr, clip_percentile)
    arr   = np.clip(arr, a_min=None, a_max=limit)

    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val > 1e-6:
        return (arr - min_val) / (max_val - min_val)
    return np.zeros_like(arr)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 統一入口
# ─────────────────────────────────────────────────────────────────────────────

_NORMALIZERS: dict[str, callable] = {
    "minmax":        minmax_normalize,
    "zscore":        zscore_normalize,
    "robust_zscore": robust_zscore_normalize,
}


def normalize(
    arr: np.ndarray,
    method: NormMethod = "minmax",
    **kwargs,
) -> np.ndarray:
    """
    統一標準化入口。

    Parameters
    ----------
    arr : np.ndarray
    method : 'minmax' | 'zscore' | 'robust_zscore'
    **kwargs
        傳遞給各標準化函式的額外參數，例如 clip_range=(-5, 5)。

    Returns
    -------
    np.ndarray

    Raises
    ------
    ValueError
        若 method 不在支援清單中。

    Examples
    --------
    >>> scores = np.random.randn(100, 10)
    >>> normalize(scores, method='minmax')
    >>> normalize(scores, method='zscore', clip_range=(-3, 3))
    >>> normalize(scores, method='robust_zscore')
    """
    if method not in _NORMALIZERS:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            f"Choose from {list(_NORMALIZERS.keys())}."
        )
    return _NORMALIZERS[method](arr, **kwargs)