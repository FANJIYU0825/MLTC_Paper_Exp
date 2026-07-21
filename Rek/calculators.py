"""
calculators.py
==============
分數計算策略（Strategy Pattern）。

所有類別實作 BaseScoreCalculator.calculate()，
統一回傳 (indices [N], scores [N, C], labels [N, C])。

公開 API
--------
BaseScoreCalculator            ← 抽象基底
StandardLossCalculator         ← per-sample BCE loss 加總 [N]，補零擴成 [N, 1]
StandardLossPerCellCalculator  ← per-cell BCE loss [N, C]
RankWeightedLossCalculator     ← Rank-Weighted Loss（論文 Eq.6-7）[N, C]
PrototypeDistanceCalculator    ← Prototype Distance (CD) [N, C]
PositiveGapCalculator          ← Positive Gap Score [N, C]

依賴
----
normalize.py  — sanitize_array
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .normalize import sanitize_array

# 延遲匯入：避免循環依賴，僅在實際呼叫時載入
try:
    from util.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def _save_logits(
    logits:   np.ndarray,
    labels:   np.ndarray,
    indices:  np.ndarray,
    loss:     np.ndarray,
    filename: str,
) -> None:
    """將 logits / labels / indices / loss 儲存為 .npz。"""
    np.savez(filename, logits=logits, labels=labels, loss=loss, indices=indices)
    logger.info(f"Logits saved to {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 抽象基底
# ─────────────────────────────────────────────────────────────────────────────
class MathUtils:
    @staticmethod
    def sanitize_array(x, fill_for_nonfinite=0.0):
        x = np.asarray(x)
        if x.size == 0: return x
        out = x.astype(float, copy=True)
        mask = ~np.isfinite(out)
        if mask.any():
            out[mask] = fill_for_nonfinite
        return out
    
class BaseScoreCalculator(ABC):
    """
    分數計算策略的抽象介面。

    所有子類別必須實作 calculate()，統一回傳三元組：
        indices : np.ndarray [N]      樣本全局索引
        scores  : np.ndarray [N, C]   每個 (樣本, 標籤) cell 的分數
        labels  : np.ndarray [N, C]   對應的原始標籤（0 / 1）
    """

    @abstractmethod
    def calculate(
        self,
        model:      torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device:     torch.device,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """回傳 (indices [N], scores [N, C], labels [N, C])"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. StandardLossCalculator
# ─────────────────────────────────────────────────────────────────────────────

class StandardLossCalculator(BaseScoreCalculator):
    """
    Per-sample BCE Loss（加總所有標籤維度）。

    回傳的 scores 為 [N, 1]，方便與其他 [N, C] 計算器統一介面。
    適合用於全局樣本篩選（非 per-label）。
    """

    def calculate(self, model, dataloader, device):
        model.eval()
        scores, indices, labels = [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Standard Loss"):
                b_ids   = batch["input_ids"].to(device)
                b_mask  = batch["attention_mask"].to(device)
                b_lbl   = batch["labels"].to(device)
                b_idx   = batch["index"]

                logits, _ = model(input_ids=b_ids, attention_mask=b_mask)
                loss = F.binary_cross_entropy_with_logits(
                    logits, b_lbl.float(), reduction="none"
                )                                   # [B, C]
                loss_sum = loss.sum(dim=1)          # [B]

                scores.extend(loss_sum.cpu().numpy())
                indices.extend(b_idx.numpy())
                labels.append(b_lbl.cpu().numpy())

        indices_np = np.array(indices)
        scores_np  = np.array(scores).reshape(-1, 1)   # [N, 1]
        labels_np  = np.vstack(labels)
        return indices_np, scores_np, labels_np


# ─────────────────────────────────────────────────────────────────────────────
# 2. StandardLossPerCellCalculator
# ─────────────────────────────────────────────────────────────────────────────

class StandardLossPerCellCalculator(BaseScoreCalculator):
    """
    Per-cell BCE Loss [N, C]。

    作為 HSMHybridPipeline 的 rel_calculator 插槽，
    可用於 HSM (a) BCE-only ablation。
    """

    def calculate(self, model, dataloader, device):
        model.eval()
        scores, indices, labels = [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Per-cell BCE Loss"):
                b_ids  = batch["input_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                b_lbl  = batch["labels"].to(device)
                b_idx  = batch["index"]

                logits, _ = model(input_ids=b_ids, attention_mask=b_mask)
                loss = F.binary_cross_entropy_with_logits(
                    logits, b_lbl.float(), reduction="none"
                )   # [B, C]

                scores.append(loss.cpu().numpy())
                indices.extend(b_idx.numpy())
                labels.append(b_lbl.cpu().numpy())

        return np.array(indices), np.vstack(scores), np.vstack(labels)


# ─────────────────────────────────────────────────────────────────────────────
# 3. RankWeightedLossCalculator
# ─────────────────────────────────────────────────────────────────────────────

class RankWeightedLossCalculator(BaseScoreCalculator):
    """
    Rank-Weighted Loss（論文 Eq.6-7）。

    權重公式（Eq.6）： W_i,j = min(log10(rank_j) + 1, θ)
    加權損失（Eq.7）： E_i,j = W_i,j × L_i,j

    Parameters
    ----------
    theta : float
        權重上界，預設 3.0。
    file_prefix : str
        中間結果 .npz 的檔名前綴。
    """

    def __init__(self, theta: float = 3.0, file_prefix: str = "dataset") -> None:
        self.theta        = theta
        self.file_prefix  = file_prefix

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _rank_weight(self, logits: torch.Tensor) -> torch.Tensor:
        """
        計算每個 cell 的 rank-based 權重。

        對每個樣本，將 logits 由大到小排名，
        再套用 W = min(log10(rank) + 1, θ)（Eq.6）。
        """
        B, C = logits.shape
        idx_sorted = torch.argsort(logits, dim=1, descending=True)

        ranks = torch.empty_like(idx_sorted, dtype=torch.float, device=logits.device)
        base  = (
            torch.arange(1, C + 1, device=logits.device, dtype=torch.float)
            .unsqueeze(0)
            .expand(B, -1)
        )
        ranks.scatter_(dim=1, index=idx_sorted, src=base)

        w      = torch.log10(ranks) + 1.0
        theta  = torch.full_like(w, float(self.theta))
        return torch.minimum(w, theta)

    # ── 公開方法 ─────────────────────────────────────────────────────────────

    def calculate(self, model, dataloader, device):
        model.eval()
        scores, indices, labels = [], [], []
        all_logits, all_losses  = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Rank Weighted Loss"):
                b_ids  = batch["input_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                b_lbl  = batch["labels"].to(device)
                b_idx  = batch["index"]

                logits, _      = model(input_ids=b_ids, attention_mask=b_mask)
                loss           = F.binary_cross_entropy_with_logits(
                    logits, b_lbl.float(), reduction="none"
                )                                       # [B, C]
                weights        = self._rank_weight(logits)
                weighted_loss  = loss * weights         # Eq.7（不做外層 clamp）

                scores.append(weighted_loss.cpu().numpy())
                all_losses.append(loss.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                indices.extend(b_idx.numpy())
                labels.append(b_lbl.cpu().numpy())

        indices_np = np.array(indices)
        scores_np  = np.vstack(scores)
        labels_np  = np.vstack(labels)

        _save_logits(
            logits=np.vstack(all_logits),
            labels=labels_np,
            indices=indices_np,
            loss=np.vstack(all_losses),
            filename=f"rank_weighted_logits_{self.file_prefix}.npz",
        )
        logger.debug(
            f"[RankWeightedLoss] scores={scores_np.shape}, indices={indices_np.shape}"
        )
        return indices_np, scores_np, labels_np


# ─────────────────────────────────────────────────────────────────────────────
# 4. PrototypeDistanceCalculator
# ─────────────────────────────────────────────────────────────────────────────

class PrototypeDistanceCalculator(BaseScoreCalculator):
    """
    Prototype Distance (CD) 計算器。

    流程
    ----
    1. _build_prototypes()：對每個 label，收集正樣本的特徵向量，
       以加權平均（Eq.9-11）建立原型。
    2. _compute_cd()：對所有樣本計算與原型的 cosine similarity，
       取負號使「距離大 = 遠離原型 = FP 嫌疑高」。
    3. 套用 sign convention：
       lab=1 → masked_dist ∈ [-1, 0]（TP 靠近 -1，FP 靠近 0）
       lab=0 → masked_dist ∈ [0, 1]（TN 靠近 0，FN 靠近 1）

    Parameters
    ----------
    args : object
        需含 args.label_size (int)。
    """

    def __init__(self, args) -> None:
        self.args           = args
        self.prototype_dict: dict[int, np.ndarray] = {}

    # ── 原型建立 ─────────────────────────────────────────────────────────────

    def _threshold_for_label(self, logit_list: list) -> float:
        """
        依論文 Eq.10-11 計算每個 label 的動態門檻 hj。
        使用 sigmoid 機率（而非 raw logits）避免量綱問題。
        """
        arr = sanitize_array(np.array(logit_list).reshape(-1))
        if arr.size == 0:
            return 0.5
        probs = 1.0 / (1.0 + np.exp(-arr))           # sigmoid → [0, 1]
        y_bar = probs.mean()
        if y_bar < 1e-8:
            return 0.5
        w  = np.maximum(1.0, probs / y_bar)           # Eq.11
        hj = float((w * probs).mean())                # Eq.10
        return hj

    def _build_prototypes(self, model, dataloader, device) -> None:
        """收集正樣本特徵並建立每個 label 的原型向量。"""
        C            = self.args.label_size
        feature_dict = {c: None for c in range(C)}
        logit_dict   = {c: None for c in range(C)}

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting features for Prototypes"):
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).float()

                logits, features = model(input_ids=ids, attention_mask=mask)

                B   = features.shape[0]
                rep = features.cpu().numpy()
                log = logits.cpu().numpy()
                lab = labels.cpu().numpy()

                is_3d = features.ndim == 3   # MltcLWAN: [B, C, H]

                for b in range(B):
                    for c in range(C):
                        if lab[b, c] != 1:
                            continue
                        vec = rep[b:b+1, c, :] if is_3d else rep[b:b+1, :]
                        lg  = log[b:b+1, c:c+1]
                        if feature_dict[c] is None:
                            feature_dict[c] = vec
                            logit_dict[c]   = lg
                        else:
                            feature_dict[c] = np.vstack((feature_dict[c], vec))
                            logit_dict[c]   = np.vstack((logit_dict[c], lg))

        H = features.shape[-1]
        for c in range(C):
            feat_list = feature_dict[c]
            log_list  = logit_dict[c] if logit_dict[c] is not None else np.zeros(1)
            thr       = self._threshold_for_label(log_list)

            if feat_list is None:
                proto = np.zeros(H)
            else:
                logs  = sanitize_array(np.array(log_list).reshape(-1), -np.inf)
                probs = 1.0 / (1.0 + np.exp(-logs))
                feats = sanitize_array(np.array(feat_list))
                mask  = probs > float(thr)
                cand  = feats[mask] if np.any(mask) else feats
                proto = np.nanmean(cand, axis=0) if cand.size > 0 else np.zeros(H)

            self.prototype_dict[c] = np.nan_to_num(proto)

    # ── CD 計算 ──────────────────────────────────────────────────────────────

    def _compute_cd(
        self,
        features: torch.Tensor,
        device:   torch.device,
    ) -> torch.Tensor:
        """
        計算每個 (樣本, label) cell 與原型的負 cosine similarity。
        回傳值域 [−1, 0]，值越負代表越靠近原型（越 clean）。
        """
        if features.ndim == 3:
            # MltcLWAN: [B, C, H]
            B, C, _ = features.shape
            dist     = torch.zeros(B, C, device=device)
            for c in range(C):
                p = self.prototype_dict.get(c)
                if p is None or np.abs(p).sum() == 0:
                    dist[:, c] = 0.0
                    continue
                feat_c  = features[:, c, :]
                proto_c = torch.tensor(p, dtype=feat_c.dtype, device=device)
                sim      = F.cosine_similarity(
                    feat_c, proto_c.unsqueeze(0).expand(B, -1), dim=1, eps=1e-8
                )
                dist[:, c] = -1.0 * torch.clamp(sim, -1.0, 1.0)
        else:
            # Mltc (CLS): [B, H]
            B, H = features.shape
            C    = len(self.prototype_dict)
            protos = []
            for c in range(C):
                p = self.prototype_dict.get(c)
                protos.append(
                    torch.tensor(p if p is not None else np.zeros(H),
                                 dtype=features.dtype, device=device)
                )
            protos_t = torch.stack(protos)              # [C, H]
            sim      = F.cosine_similarity(
                features.unsqueeze(1), protos_t.unsqueeze(0), dim=2, eps=1e-8
            )                                           # [B, C]
            dist     = -1.0 * torch.clamp(sim, -1.0, 1.0)

            # 無原型的 label 距離設為 0（不貢獻分數）
            has_proto = (protos_t.abs().sum(dim=1) > 0).float().unsqueeze(0)
            dist      = dist * has_proto

        return dist   # [B, C]

    # ── 公開方法 ─────────────────────────────────────────────────────────────

    def calculate(self, model, dataloader, device):
        self._build_prototypes(model, dataloader, device)

        model.eval()
        scores, indices, labels = [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Prototype Distance"):
                b_lbl = batch["labels"].to(device)
                b_idx = batch["index"]

                _, features = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                dists = self._compute_cd(features, device)   # [B, C]

                lab_np  = b_lbl.cpu().numpy()
                dist_np = dists.cpu().numpy()

                # Sign convention（工程處理，使 GMM clean/noisy 方向一致）:
                #   lab=1 : dist * 1  → [-1, 0]  (TP → -1, FP → 0)
                #   lab=0 : dist * -1 → [0,  1]  (TN →  0, FN → 1)
                masked = dist_np * (2 * lab_np - 1)
                scores.append(masked)
                indices.extend(b_idx.numpy())
                labels.append(lab_np)

        return np.array(indices), np.vstack(scores), np.vstack(labels)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PositiveGapCalculator
# ─────────────────────────────────────────────────────────────────────────────

class PositiveGapCalculator(BaseScoreCalculator):
    """
    Positive Gap Score。

    定義：margin[b, c] = max(logits[b, :]) − logits[b, c]
        - margin 大 → logit[c] 遠低於最高分 label → 模型不看好此 label → FP 嫌疑高
        - margin 小 → logit[c] 接近最高分 → 模型幾乎也認為此 label = 1

    Sign convention（與 CD 一致）：
        masked = margin × (2 × lab − 1)
        lab=1: margin 大 → score 高 → FP 嫌疑高
        lab=0: margin 小 → 翻轉後 score 高 → FN 嫌疑高
    """

    def calculate(self, model, dataloader, device):
        model.eval()
        scores, indices, labels = [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Positive Gap"):
                b_ids  = batch["input_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                b_lbl  = batch["labels"].to(device)
                b_idx  = batch["index"]

                logits, _ = model(input_ids=b_ids, attention_mask=b_mask)
                max_logit  = logits.max(dim=1, keepdim=True).values   # [B, 1]
                margin     = max_logit - logits                        # [B, C]

                lab_np    = b_lbl.cpu().numpy()
                margin_np = margin.cpu().numpy()
                masked    = margin_np * (2 * lab_np - 1)

                scores.append(masked)
                indices.extend(b_idx.numpy())
                labels.append(lab_np)

        indices_np = np.array(indices)
        scores_np  = np.vstack(scores)
        labels_np  = np.vstack(labels)
        logger.debug(f"[PositiveGapCalculator] scores={scores_np.shape}")
        return indices_np, scores_np, labels_np