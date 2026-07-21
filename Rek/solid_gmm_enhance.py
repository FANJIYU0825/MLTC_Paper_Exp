
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture as GMM
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from util.logger import logger
from .experence import (debug_plot_gmm, visualize_gmm, visualize_2d_gmm_candidates,
                        visualize_gmm_cluster_purity_k, visualize_1d_gmm_features)
import pickle


def save_logits(logits, labels,index,loss, filename):
    """將 logits、labels 和 indices 儲存為 .npz 檔案"""
    np.savez(filename, logits=logits, labels=labels,loss = loss, indices=index)
    print(f"Logits saved to {filename}")
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

    @staticmethod
    def robust_normalize(score_array, clip_percentile=99.0):
        """抗極端值的歸一化 (Robust Normalization)"""
        arr = np.array(score_array)
        limit = np.percentile(arr, clip_percentile)
        arr_clipped = np.clip(arr, a_min=None, a_max=limit)
        min_val = arr_clipped.min()
        max_val = arr_clipped.max()

        if max_val - min_val > 1e-6:
            return (arr_clipped - min_val) / (max_val - min_val)
        return np.zeros_like(arr)

    @staticmethod
    def zscore_normalize_columnwise(matrix, clip_range=None):
        """
        Z-Score 標準化 (column-wise)
        將每一個 column (標籤) 轉換為 z-score: (x - mean) / std

        Args:
            matrix: [N, C] or [N] 型態的分數矩陣
            clip_range: Optional tuple (min, max) 限制 z-score 範圍，例如 (-5, 5)

        Returns:
            標準化後的矩陣，每個 column 的 mean≈0, std≈1
        """
        out = matrix.astype(float, copy=True)

        # 保護：確保至少是 2D，才能做 column-wise 操作
        squeezed = out.ndim == 1
        if squeezed:
            out = out.reshape(-1, 1)

        # 計算每個 column 的均值和標準差
        mean_vals = out.mean(axis=0)  # shape: (C,)
        std_vals = out.std(axis=0)    # shape: (C,)

        # 避免除以 0 (如果該標籤所有分數都一樣)
        std_vals[std_vals < 1e-8] = 1.0

        # Broadcasting 運算: (N, C) - (C,) / (C,)
        out = (out - mean_vals) / std_vals

        # Optional: 限制極端值範圍
        if clip_range is not None:
            out = np.clip(out, clip_range[0], clip_range[1])

        # 還原原本的形狀
        if squeezed:
            out = out.squeeze(1)

        return out

    @staticmethod
    def robust_zscore_normalize_columnwise(matrix, clip_range=None):
        """
        Robust Z-Score 標準化 (使用中位數和 MAD)
        對極端值更穩健的版本

        Args:
            matrix: [N, C] or [N] 型態的分數矩陣
            clip_range: Optional tuple (min, max) 限制 z-score 範圍

        Returns:
            標準化後的矩陣
        """
        out = matrix.astype(float, copy=True)

        squeezed = out.ndim == 1
        if squeezed:
            out = out.reshape(-1, 1)

        # 使用中位數代替均值
        median_vals = np.median(out, axis=0)

        # 計算 MAD (Median Absolute Deviation)
        mad_vals = np.median(np.abs(out - median_vals), axis=0)

        # 轉換為標準差估計 (1.4826 是常態分佈下的轉換係數)
        scale = 1.4826 * mad_vals
        scale[scale < 1e-8] = 1.0

        out = (out - median_vals) / scale

        if clip_range is not None:
            out = np.clip(out, clip_range[0], clip_range[1])

        if squeezed:
            out = out.squeeze(1)

        return out
    @staticmethod
    # def minmax_normalize_columnwise(matrix):
    #     """
    #     label_wise normalize_array
        
    #     """
    #     # 避免修改原始數據
    #     out = matrix.astype(float, copy=True)
    #     # 針對每一行 (Column) 找 min 和 max
    #     min_vals = out.min(axis=0)
    #     max_vals = out.max(axis=0)
        
    #     # 避免除以 0 (如果該標籤所有分數都一樣)
    #     range_vals = max_vals - min_vals
    #     range_vals[range_vals == 0] = 1e-8 
        
    #     # Broadcasting 運算: (N, C) - (C,) / (C,)
    #     out = (out - min_vals) / range_vals
    #     return out
    def minmax_normalize_columnwise(matrix):
        out = matrix.astype(float, copy=True)
        
        # 保護：確保至少是 2D，才能做 column-wise 操作
        squeezed = out.ndim == 1
        if squeezed:
            out = out.reshape(-1, 1)
        
        min_vals = out.min(axis=0)
        max_vals = out.max(axis=0)
        
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-8
        
        out = (out - min_vals) / range_vals
        
        # 還原原本的形狀
        if squeezed:
            out = out.squeeze(1)
        
        return out

class BaseScoreCalculator(ABC):
    """計算分數的基礎介面 (Strategy Pattern)"""
    @abstractmethod
    def calculate(self, model, dataloader, device):
        """回傳 (indices, scores)"""
        pass

class BaseNoiseFilter(ABC):
    """篩選雜訊的基礎介面"""
    @abstractmethod
    def filter(self, scores, indices):
        """回傳 (clean_indices, noisy_indices)"""
        pass

class StandardLossCalculator(BaseScoreCalculator):
    """計算原始 BCE Loss"""
    def calculate(self, model, dataloader, device):
        model.eval()
        scores = []
        indices = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Standard Loss"):
                b_input_ids = batch['input_ids'].to(device)
                b_attn_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                b_idx = batch['index']

                logits, _ = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.float(), reduction='none')
                loss_sum = loss.sum(dim=1)
                
                scores.extend(loss_sum.cpu().numpy())
                indices.extend(b_idx.numpy())
                
        return np.array(indices), np.array(scores)

class RankWeightedLossCalculator(BaseScoreCalculator):
    """計算 Rank-based Weighted Loss"""
    def __init__(self, theta=3.0, file_prefix='aapd_mltc'):
        self.theta = theta
        self._file_prefix = file_prefix
    
    def _calculate_rank_weight(self, logits):
        B, C = logits.shape
        idx_sorted = torch.argsort(logits, dim=1, descending=True)
        
        ranks = torch.empty_like(idx_sorted, dtype=torch.float, device=logits.device)
        base = torch.arange(1, C + 1, device=logits.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        ranks.scatter_(dim=1, index=idx_sorted, src=base)
        
        w = torch.log10(ranks) + 1.0
        theta_tensor = torch.full_like(w, float(self.theta))
        w = torch.minimum(w, theta_tensor)
        return w

    def calculate(self, model, dataloader, device):
        model.eval()
        scores = []
        indices = []
        labels = []
        all_losses = []  # 儲存 Loss
        all_logits = []  # 儲存 Logits
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Rank Weighted Loss"):
                b_input_ids = batch['input_ids'].to(device)
                b_attn_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                b_idx = batch['index']
                
                logits, _ = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.float(), reduction='none')
                
                weights = self._calculate_rank_weight(logits=logits)

                # [Fix 1] 對齊 paper Eq (7)：E_i,j = W × L，乘積不再 clamp。
                # 原版的外層 clamp 會把雜訊樣本最強的 ~1.2% 訊號壓到 θ，
                # 導致 GMM「雜訊群」mean 卡在 θ=3，無法跟乾淨群拉開。
                # W 本身的 clamp 仍保留在 _calculate_rank_weight (Eq 6)。
                #
                # ── 原版（保留以便回退）─────────────────────────────────
                # weighted_loss = torch.clamp(loss * weights, max=self.theta)
                # ───────────────────────────────────────────────────────
                weighted_loss = loss * weights

                scores.append(weighted_loss.cpu().numpy()) 
                
                # --- 修正這裡 ---
                all_losses.append(loss.cpu().numpy())    # 把 loss 加到 all_losses
                all_logits.append(logits.cpu().numpy())  # 把 logits 加到 all_logits
                # ----------------
                
                indices.extend(b_idx.numpy())
                labels.append(b_labels.cpu().numpy())
                
        # 轉換與堆疊
        indices_np = np.array(indices)
        scores_np = np.vstack(scores)
        labels_np = np.vstack(labels)
        
        logits_np = np.vstack(all_logits)
        logss_np = np.vstack(all_losses)  # 現在這裡不會報錯了，因為裡面有資料
        save_logits(logits_np, labels_np, indices_np,logss_np, filename=f"rank_weighted_logits_{self._file_prefix}.npz")
        print(f"Debug Calculator Output:")
        print(f"  - Scores shape: {scores_np.shape}") # matric(54840, 54)
        print(f"  - Indices shape: {indices_np.shape}") #  (54840,)
        
        return indices_np, scores_np, labels_np

class PrototypeDistanceCalculator(BaseScoreCalculator):

    """計算 Prototype Distance (CD)"""
    def __init__(self, args):
        self.args = args
        self.prototype_dict = {}

    def _get_threshold(self, logit_list):
        arr = MathUtils.sanitize_array(np.array(logit_list).reshape(-1))
        if arr.size == 0: return 0.5
        # 論文 Eq.(10-11): 使用 sigmoid 機率而非 raw logits
        probs = 1.0 / (1.0 + np.exp(-arr))  # sigmoid → [0, 1]
        y_bar = probs.mean()
        if y_bar < 1e-8: return 0.5
        w = np.maximum(1.0, probs / y_bar)   # Eq.(11): wi = max(1, Ŷi,j / Ȳj)
        hj = float((w * probs).mean())        # Eq.(10): hj = weighted avg
        return hj

    def _build_prototypes(self, model, dataloader, device):
        feature_dict = {idx: None for idx in range(self.args.label_size)}
        logit_dict = {idx: None for idx in range(self.args.label_size)}

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting features for Prototypes"):
                # 這裡簡化參數傳遞，視你的 Dataset 而定
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).float()

                # 假設 forward 回傳 (logits, features)
                logits, features = model(input_ids=input_ids, attention_mask=attention_mask)

                # 支援兩種 features 格式
                if features.ndim == 3:
                    # MltcLWAN: [B, L, H]
                    B, C, H = features.shape
                else:
                    # Mltc (CLS): [B, H]
                    B, H = features.shape
                    C = self.args.label_size

                rep_np = features.cpu().numpy()
                log_np = logits.cpu().numpy()
                lab_np = labels.cpu().numpy()

                for b in range(B):
                    for c in range(C):
                        if lab_np[b, c] == 1:
                            # 根據 features 維度取出對應的 vector
                            if features.ndim == 3:
                                # MltcLWAN: 取 label c 的專屬 feature
                                vec = rep_np[b:b+1, c, :]  # [1, H]
                            else:
                                # Mltc: 取共用的 [CLS] feature
                                vec = rep_np[b:b+1, :]  # [1, H]

                            lg = log_np[b:b+1, c:c+1]
                            if feature_dict[c] is None:
                                feature_dict[c] = vec
                                logit_dict[c] = lg
                            else:
                                feature_dict[c] = np.vstack((feature_dict[c], vec))
                                logit_dict[c] = np.vstack((logit_dict[c], lg))
        
        # Aggregate
        for key in feature_dict.keys():
            feat_list = feature_dict[key]
            log_list = logit_dict[key]
            if log_list is None: log_list = np.zeros(1)  # fix: 不用 random，語意為「無資料」
            thr = self._get_threshold(log_list)

            # Get single prototype
            if feat_list is None:
                proto = np.zeros(features.shape[-1])  # fix: shape[-1] 相容 2D/3D features
            else:
                logs = MathUtils.sanitize_array(np.array(log_list).reshape(-1), -np.inf)
                probs = 1.0 / (1.0 + np.exp(-logs))  # fix: sigmoid → 與 thr 同單位 (Eq.9)
                feats = MathUtils.sanitize_array(np.array(feat_list))
                mask = probs > float(thr)
                if np.any(mask):
                    cand = feats[mask]
                    proto = np.nanmean(cand, axis=0) if cand.size > 0 else np.nanmean(feats, axis=0)
                else:
                    proto = np.nanmean(feats, axis=0)

            self.prototype_dict[key] = np.nan_to_num(proto)

    def _compute_vectorized_cd(self, features, device):
        """
        計算 Prototype Distance (CD)
        範圍限制在 0 ~ -1 (越負代表越接近原型)
        """
        if features.ndim == 3:
            # ─── MltcLWAN: [B, L, H] ───
            B, L, H = features.shape
            C = len(self.prototype_dict)
            dist = torch.zeros(B, C, device=device)

            for l in range(C):
                feat_l = features[:, l, :]
                p = self.prototype_dict.get(l)
                
                if p is None or np.abs(p).sum() == 0:
                    # 無原型時，距離為 0 (代表最不相似/最遠)
                    dist[:, l] = 0.0
                else:
                    proto_l = torch.tensor(p, dtype=feat_l.dtype, device=device)
                    sim = F.cosine_similarity(feat_l, proto_l.unsqueeze(0).expand(B, -1), dim=1, eps=1e-8)
                    
                    # 論文邏輯: -cos，且限制在 0 到 -1
                    dist[:, l] = -1.0 * torch.clamp(sim, min=-1.0, max=1.0)

        else:
            # ─── Mltc (CLS): [B, H] ───
            B, H = features.shape
            C = len(self.prototype_dict)
            protos = []
            for i in range(C):
                p = self.prototype_dict.get(i)
                if p is None: p = np.zeros(H)
                protos.append(torch.tensor(p, dtype=features.dtype, device=device))

            protos_tensor = torch.stack(protos)  # [C, H]
            sim = F.cosine_similarity(features.unsqueeze(1), protos_tensor.unsqueeze(0), dim=2, eps=1e-8)
            
            # 限制範圍並取負號
            dist = -1.0 * torch.clamp(sim, min=-1.0, max=1.0)

            # Handle empty prototypes: 沒有原型的類別距離設為 0
            has_proto = (protos_tensor.abs().sum(dim=1) > 0).float().unsqueeze(0)
            dist = dist * has_proto + (1.0 - has_proto) * 0.0

        return dist

    def calculate(self, model, dataloader, device):
        self._build_prototypes(model, dataloader, device)
        
        model.eval()
        scores = []
        indices = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Prototype Distance"):
                b_labels = batch['labels'].to(device)
                b_idx = batch['index']
                
                # 取得原始特徵並計算 dist (範圍 0 ~ -1)
                _, features = model(input_ids=batch['input_ids'].to(device), 
                                    attention_mask=batch['attention_mask'].to(device))
                dists = self._compute_vectorized_cd(features, device)  # [B, C]
                
                lab_np = b_labels.cpu().numpy()
                dist_np = dists.cpu().numpy()
                
                # ─── 關鍵翻轉邏輯 ───
                # lab=1 (正類): dist * (1)  -> 範圍 [-1, 0] (TP 靠近 -1, FP 靠近 0)
                # lab=0 (負類): dist * (-1) -> 範圍 [0, 1]  (TN 靠近 0, FN 靠近 1)
                # 註：雖然不在 paper Eq (10) 裡，但這是為了讓負樣本側
                # GMM 的「argmin=clean」convention 也能成立的工程處理。
                masked_dists = dist_np * (2 * lab_np - 1)

                scores.append(masked_dists)
                indices.extend(b_idx.numpy())
                labels.append(lab_np)

        return np.array(indices), np.vstack(scores), np.vstack(labels)
class PositiveGapCalculator(BaseScoreCalculator):
    """
    計算 Positive Gap Score（方案 A）

    定義：margin[b, c] = max(logits[b, :]) - logits[b, c]
        - margin 大 → logit[c] 遠低於最高分 label → 模型不看好此 label
        - margin 小 → logit[c] 接近最高分 → 模型幾乎也認為此 label = 1

    Sign convention（與 CD 一致）：
        masked_margin = margin * (2 * lab - 1)
        - lab=1（正樣本）：margin 大 → score 高 → FP 嫌疑高
        - lab=0（負樣本）：margin 小 → 翻轉後 score 高 → FN 嫌疑高
    """

    def calculate(self, model, dataloader, device):
        model.eval()
        scores = []
        indices = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Positive Gap"):
                b_input_ids = batch['input_ids'].to(device)
                b_attn_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                b_idx = batch['index']

                logits, _ = model(input_ids=b_input_ids, attention_mask=b_attn_mask)

                # margin[b, c] = max(logits[b, :]) - logits[b, c]
                max_logit = logits.max(dim=1, keepdim=True).values  # [B, 1]
                margin = max_logit - logits                          # [B, C]

                lab_np = b_labels.cpu().numpy()
                margin_np = margin.cpu().numpy()

                # lab=1: margin 大 = FP 嫌疑高（保留正號）
                # lab=0: margin 小 = FN 嫌疑高（翻轉負號）
                masked_margin = margin_np * (2 * lab_np - 1)

                scores.append(masked_margin)
                indices.extend(b_idx.numpy())
                labels.append(lab_np)

        indices_np = np.array(indices)
        scores_np = np.vstack(scores)
        labels_np = np.vstack(labels)
        print(f"[PositiveGapCalculator] Scores shape: {scores_np.shape}")
        return indices_np, scores_np, labels_np


class HSMHybridPipeline:
    """
    專門處理 HSM 這種需要結合兩種分數的複雜流程
    這是一個更高階的 Orchestrator
    """
    def __init__(self, rel_calculator, cd_calculator, alpha=0.5,
                 gap_calculator=None, beta=1.0):
        self.rel_calc = rel_calculator
        self.cd_calc = cd_calculator
        self.gap_calc = gap_calculator

        self.alpha = alpha
        # beta: CD 在 (1-alpha) 部分的佔比
        # beta=1.0 → 純 CD（無 margin，退化為原始 HSM）
        # beta=0.0 → 純 Margin（無 CD）
        self.beta = beta




    def run_score_only(self, model, dataloader, device, encoder_name='mltc',
                       normalization='minmax', clip_range=None, dataset_name='aapd'):
        """
        只計算並融合分數，不進行篩選/校正 (為了外部參數搜尋用)

        Args:
            model: 訓練好的模型
            dataloader: 數據加載器
            device: 運算設備
            encoder_name: 編碼器名稱 (用於檔案命名)
            normalization: 標準化方法 'minmax' | 'zscore' | 'robust_zscore'
            clip_range: z-score 的裁剪範圍，例如 (-5, 5)，僅在 zscore 模式下有效
        """
        print("Pipeline: Calculating Scores Only...")

        # 1. 計算原始分數
        idx1, rel_scores, labels = self.rel_calc.calculate(model, dataloader, device)
        idx_c, cd_scores, labels = self.cd_calc.calculate(model, dataloader, device)

        # 1b. 若有 gap calculator，也一起計算
        if self.gap_calc is not None:
            idx_m, margin_scores, _ = self.gap_calc.calculate(model, dataloader, device)

        # 2. 根據選擇的方法進行標準化
        def _normalize(arr):
            if normalization == 'zscore':
                return MathUtils.zscore_normalize_columnwise(arr, clip_range=clip_range)
            elif normalization == 'robust_zscore':
                return MathUtils.robust_zscore_normalize_columnwise(arr, clip_range=clip_range)
            elif normalization == 'minmax':
                return MathUtils.minmax_normalize_columnwise(arr)
            else:
                raise ValueError(f"Unknown normalization method: {normalization}")

        rel_norm = _normalize(rel_scores)
        cd_norm = _normalize(cd_scores)
        print(f"  Normalization: {normalization} (clip_range={clip_range})")

        # 3. 儲存中間結果
        suffix = f"_{normalization}" if normalization != 'minmax' else ""
        save_path = f"data/evaluation_results_{dataset_name}_{encoder_name}{suffix}.npz"

        save_kwargs = dict(
            idx_rel=idx1,
            rel_scores=rel_scores,
            rel_norm=rel_norm,
            idx_cd=idx_c,
            cd_scores=cd_scores,
            cd_norm=cd_norm,
            labels=labels,
            normalization=normalization,
            alpha=self.alpha,
            beta=self.beta,
        )
        if self.gap_calc is not None:
            margin_norm = _normalize(margin_scores)
            save_kwargs['margin_scores'] = margin_scores
            save_kwargs['margin_norm'] = margin_norm

        np.savez_compressed(save_path, **save_kwargs)
        print(f"  Scores saved to: {save_path}")

        # 4. 融合
        # HSM = alpha * REL + (1-alpha) * (beta * CD + (1-beta) * Margin)
        if self.gap_calc is not None:
            cd_part = self.beta * cd_norm + (1 - self.beta) * margin_norm
            print(f"  Fusion: alpha={self.alpha}, beta={self.beta} (REL + CD + Positive Gap)")
        else:
            cd_part = cd_norm
            print(f"  Fusion: alpha={self.alpha} (REL + CD, no Positive Gap)")

        hsm_scores = self.alpha * rel_norm + (1 - self.alpha) * cd_part

        return hsm_scores, labels, idx1

    def run_score_separately(self, model, dataloader, device, encoder_name='mltc',
                             normalization='minmax', clip_range=None, dataset_name='aapd'):
        """
        計算並分別回傳各項原始分數（不融合），供兩階段校正器使用。

        Returns:
            rel_norm:  [N, C] 標準化 REL 分數
            cd_norm:   [N, C] 標準化 CD 分數
            gap_norm:  [N, C] 標準化 Positive Gap 分數
            labels:    [N, C] 原始標籤
            indices:   [N]    樣本索引
        """
        if self.gap_calc is None:
            raise ValueError("run_score_separately 需要 gap_calculator，請在初始化時傳入。")

        print("Pipeline: Calculating Scores Separately (for TwoStage)...")

        idx1, rel_scores, labels = self.rel_calc.calculate(model, dataloader, device)
        idx_c, cd_scores, _      = self.cd_calc.calculate(model, dataloader, device)
        idx_m, gap_scores, _     = self.gap_calc.calculate(model, dataloader, device)

        def _normalize(arr):
            if normalization == 'zscore':
                return MathUtils.zscore_normalize_columnwise(arr, clip_range=clip_range)
            elif normalization == 'robust_zscore':
                return MathUtils.robust_zscore_normalize_columnwise(arr, clip_range=clip_range)
            elif normalization == 'minmax':
                return MathUtils.minmax_normalize_columnwise(arr)
            else:
                raise ValueError(f"Unknown normalization method: {normalization}")

        rel_norm = _normalize(rel_scores)
        cd_norm  = _normalize(cd_scores)
        gap_norm = _normalize(gap_scores)

        print(f"  [run_score_separately] rel:{rel_scores.shape}, cd:{cd_scores.shape}, gap:{gap_scores.shape}")
        # 回傳 raw scores 供 TwoStage 使用（保留原始 scale，不壓縮）
        # 同時回傳 norm 供外部推導 hsm_scores 使用
        return rel_scores, cd_scores, gap_scores, rel_norm, cd_norm, gap_norm, labels, idx1


class RELOnlyCorrector:
    """
    Stage 1 only: 對每個 label，取 label=1 樣本中 REL 最高的 top_ratio 直接翻轉為 0。
    作為 TwoStageRELFPCorrector 的基線，衡量 Stage 1 單獨效果。
    """

    def __init__(self, top_ratio: float = 0.01):
        self.top_ratio = top_ratio

    def correct(self, rel_scores: np.ndarray, labels: np.ndarray, **kwargs) -> np.ndarray:
        rel    = np.asarray(rel_scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        N, C   = labels.shape
        corrected = labels.astype(float).copy()

        total_flipped = 0
        for c in range(C):
            pos_indices = np.where(labels[:, c] == 1)[0]
            if len(pos_indices) < 4:
                continue
            top_k = max(1, int(len(pos_indices) * self.top_ratio))
            local_top    = np.argsort(rel[pos_indices, c])[-top_k:]
            cand_indices = pos_indices[local_top]
            corrected[cand_indices, c] = 0.0
            total_flipped += len(cand_indices)
            logger.info(
                f"  [RELOnly Label {c}] flipped={len(cand_indices)}"
                f" ({len(cand_indices)/len(pos_indices):.1%} of positives)"
            )

        logger.info(f"[RELOnlyCorrector] Done. total_flipped={total_flipped}")
        return corrected


class TwoStageRELFPCorrector:
    """
    兩階段 FP 校正器（僅針對 label=1 的樣本）:

    Stage 1 — REL 篩選 (per-label):
        對每個 label c，取所有 label=1 樣本中 REL 分數最高的 top_ratio 作為 FP 候選池。
        REL 高 → 模型對此 label 損失大 → 標記為 1 但模型不認同 → FP 嫌疑高。

    Stage 2 — 2D GMM(CD, Gap):
        對候選池以 (CD, Positive Gap) 兩個特徵 fit 2-component GMM。
        CD 與 Gap 比 REL 更接近 Gaussian 分佈，更符合 GMM 前提假設。
        以 epsilon-band 輸出校正後的標籤（soft label）。
    """

    def __init__(self, top_ratio: float = 0.05, epsilon: float = 0.05, n_components: int = 2,
                 feature_mode: str = '2d'):
        """
        Args:
            top_ratio:    Stage 1 REL top-N% 直接翻轉比例
            epsilon:      epsilon-band 校正邊界
            n_components: Stage 2 GMM 群數 (2..5)
            feature_mode: Stage 2 GMM 特徵模式
                '2d'  — 原始 2D (−CD, Gap) 聯合 GMM（預設）
                'cd'  — 只用 −CD 做 1D GMM
                'gap' — 只用 Positive Gap 做 1D GMM
        """
        self.top_ratio    = top_ratio
        self.epsilon      = epsilon
        self.n_components = n_components
        self.feature_mode = feature_mode   # '2d' | 'cd' | 'gap'

    # ─────────────────────────────────────────────────────────────
    # 1D GMM helpers
    # ─────────────────────────────────────────────────────────────

    def _run_1d_gmm(self, scores_1d: np.ndarray) -> dict:
        """
        在 1D 分數上 fit n_components-GMM。
        排序方式：mean 升序，idx=0 最小(clean)，idx=-1 最大(noisy)。
        回傳與 _run_2d_gmm 相同格式的 prob_dict。
        """
        data = scores_1d.reshape(-1, 1)
        gmm = GMM(n_components=self.n_components, max_iter=200, tol=1e-3,
                  reg_covar=1e-4, random_state=0)
        gmm.fit(data)

        sorted_idx = np.argsort(gmm.means_.flatten())
        probs_all  = gmm.predict_proba(data)

        result = {
            'clean_probs': probs_all[:, sorted_idx[0]],
            'noisy_probs': probs_all[:, sorted_idx[-1]],
        }
        mid_indices = sorted_idx[1:-1]
        if len(mid_indices) == 1:
            result['mid_probs'] = probs_all[:, mid_indices[0]]
        else:
            for i, idx in enumerate(mid_indices):
                result[f'mid_probs_{i}'] = probs_all[:, idx]
        return result

    def _fit_2d_gmm(self, cd_vals: np.ndarray, gap_vals: np.ndarray) -> np.ndarray:
        """
        在 (CD, Gap) 2D 空間 fit 2-component GMM。

        CD masked for label=1 範圍 [-1, 0]：
            TP → close to -1（靠近原型，cosine similarity 高）
            FP → close to  0（遠離原型，cosine similarity 低）
        取負號後 [0, 1]：數值越大 = FP 嫌疑越高，方向與 Gap 一致。

        Gap masked for label=1 為正值：
            TP → 小（模型對此 label 有把握）
            FP → 大（模型更偏好其他 label）

        Returns:
            prob_clean: [n] 每個候選樣本屬於「乾淨 TP 群」的機率
        """
        x1 = cd_vals   # 翻轉 CD：大 = FP 嫌疑高（與 Gap 方向一致）
        x2 =  gap_vals  # Gap：大 = FP 嫌疑高

        # 不做 minmax：pipeline 已做過 normalize，
        # GMM 假設 Gaussian 分佈，minmax 會破壞真實分佈形狀。
        X = np.column_stack([x1, x2])    # [n, 2]

        gmm = GMM(n_components=2, max_iter=200, tol=1e-3,
                  reg_covar=1e-4, random_state=0)
        gmm.fit(X)

        # 均值範數較小的 component = clean（TP）群
        mean_norms = np.linalg.norm(gmm.means_, axis=1)
        clean_idx  = int(mean_norms.argmin())
        prob_clean = gmm.predict_proba(X)[:, clean_idx]
        return prob_clean

    def _apply_epsilon_band(self, prob_clean: np.ndarray, epsilon: float) -> np.ndarray:
        """
        epsilon-band soft label:
            prob_clean > 0.5 + ε  →  1.0（確定是 TP，保留）
            prob_clean < 0.5 - ε  →  0.0（確定是 FP，翻轉）
            中間帶               →  prob_clean（soft label，不確定）
        """
        prob_clean = np.nan_to_num(prob_clean, nan=0.5)
        return np.where(prob_clean > 0.5 + epsilon, 1.0,
               np.where(prob_clean < 0.5 - epsilon, 0.0,
                        prob_clean))

    def _run_2d_gmm(self, cd_vals: np.ndarray, gap_vals: np.ndarray) -> dict:
        """
        鏡像 LabelRefiner._run_gmm，但輸入是 2D (−CD, Gap) 特徵。
        Component 依 mean norm 排序：idx=0 最小(clean)，idx=-1 最大(noisy)。
        回傳與 LabelRefiner 相同格式的 prob_dict。
        """
        x1 = cd_vals   # 翻轉 CD，使 FP 嫌疑高 = 大值
        x2 = gap_vals
        X  = np.column_stack([x1, x2])   # [n, 2]

        gmm = GMM(n_components=self.n_components, max_iter=200, tol=1e-3,
                  reg_covar=1e-4, random_state=0)
        gmm.fit(X)

        # 排序：norm 小 = clean，norm 大 = noisy（同 LabelRefiner mean 排序）
        sorted_idx  = np.argsort(np.linalg.norm(gmm.means_, axis=1))
        probs_all   = gmm.predict_proba(X)

        result = {
            'clean_probs': probs_all[:, sorted_idx[0]],
            'noisy_probs': probs_all[:, sorted_idx[-1]],
        }
        mid_indices = sorted_idx[1:-1]
        if len(mid_indices) == 1:
            result['mid_probs'] = probs_all[:, mid_indices[0]]
        else:
            for i, idx in enumerate(mid_indices):
                result[f'mid_probs_{i}'] = probs_all[:, idx]
        return result

    def _apply_2d_correction_logic(self, prob_dict: dict, epsilon: float) -> np.ndarray:
        """
        鏡像 LabelRefiner._apply_correction_logic，Stage2 候選池全為 label=1 (FP context)。
        target_y=1.0 固定（Stage2 候選都是 positive samples）。
        2-comp : epsilon-band（同原本 _apply_epsilon_band）
        3-comp : argmax → clean=1.0 / mid=0.5 / noisy=0.0
        4-comp : gradient → clean=1.0 / mid-clean=0.7 / mid-noisy=0.3 / noisy=0.0
        5-comp : gradient → 1.0 / 0.75 / 0.5 / 0.25 / 0.0
        """
        n = self.n_components
        if n == 2:
            probs = np.nan_to_num(prob_dict['clean_probs'], nan=0.5)
            return np.where(probs > 0.5 + epsilon, 1.0,
                   np.where(probs < 0.5 - epsilon, 0.0,
                            probs))

        # 3+ comp：組成 stack，取 argmax 後對應 gradient 值
        if n == 3:
            stack = np.vstack([prob_dict['clean_probs'],
                               prob_dict['mid_probs'],
                               prob_dict['noisy_probs']])
            gradient = [1.0, 0.5, 0.0]
        elif n == 4:
            stack = np.vstack([prob_dict['clean_probs'],
                               prob_dict['mid_probs_0'],
                               prob_dict['mid_probs_1'],
                               prob_dict['noisy_probs']])
            gradient = [1.0, 0.7, 0.3, 0.0]
        else:  # 5
            stack = np.vstack([prob_dict['clean_probs'],
                               prob_dict['mid_probs_0'],
                               prob_dict['mid_probs_1'],
                               prob_dict['mid_probs_2'],
                               prob_dict['noisy_probs']])
            gradient = [1.0, 0.75, 0.5, 0.25, 0.0]

        winners = np.argmax(stack, axis=0)
        out = np.zeros(len(winners), dtype=float)
        for i, val in enumerate(gradient):
            out[winners == i] = val
        return out

    def correct(self, rel_scores: np.ndarray, cd_scores: np.ndarray,
                gap_scores: np.ndarray, labels: np.ndarray,
                args, y_true: np.ndarray = None) -> np.ndarray:
        """
        Args:
            rel_scores: [N, C] 標準化 REL 分數（值越大代表 FP 嫌疑越高）
            cd_scores:  [N, C] 標準化 CD 分數（masked，label=1 → 值在 [-1, 0]）
            gap_scores: [N, C] 標準化 Gap 分數（masked，label=1 → 正值）
            labels:     [N, C] 原始 noisy 標籤（0 or 1）
            args:       含 args.epsilon
            y_true:     [N, C] 真實標籤（可選，用於視覺化 TP/FP overlap）
        Returns:
            corrected:  [N, C] float 校正後標籤
        """
        rel    = np.asarray(rel_scores, dtype=float)
        cd     = np.asarray(cd_scores,  dtype=float)
        gap    = np.asarray(gap_scores, dtype=float)
        labels = np.asarray(labels,     dtype=int)

        N, C      = labels.shape
        corrected = labels.astype(float).copy()
        epsilon   = getattr(args, 'epsilon', self.epsilon)

        total_candidates = 0
        total_flipped    = 0
        total_soft       = 0

        for c in range(C):
            pos_indices = np.where(labels[:, c] == 1)[0]   # 全局 row index

            if len(pos_indices) < 4:
                continue

            # ── Stage 1: REL top-N% 直接翻轉 ─────────────────────────
            rel_c = rel[pos_indices, c]
            top_k = int(len(pos_indices) * self.top_ratio)

            if top_k == 0:
                continue

            # argsort 升序，取最後 top_k 個（REL 最高 = 最可疑）
            local_top    = np.argsort(rel_c)[-top_k:]
            cand_indices = pos_indices[local_top]    # 全局 index

            # 直接翻轉：1-label（FP:1→0, FN:0→1，方向自動正確）
            corrected[cand_indices, c] = 1.0 - labels[cand_indices, c]
            total_flipped += len(cand_indices)

            logger.info(
                f"  [TwoStage Label {c}] Stage1 flipped={len(cand_indices)}"
            )

            # ── Stage 2: 剩下 95% → 2D GMM(CD, Gap) ─────────────────
            local_rest    = np.argsort(rel_c)[:-top_k]   # REL 較低的其餘樣本
            rest_indices  = pos_indices[local_rest]

            if len(rest_indices) < 4:   # GMM 需要至少 4 個點
                continue

            total_candidates += len(rest_indices)

            cd_c  = cd[rest_indices, c]
            gap_c = gap[rest_indices, c]
            true_c = y_true[rest_indices, c] if y_true is not None else None

            # 記錄候選池 FP 混入率
            if true_c is not None:
                n_fp_in_pool = int((true_c == 0).sum())
                n_tp_in_pool = int((true_c == 1).sum())
                fp_ratio = n_fp_in_pool / len(true_c) if len(true_c) > 0 else 0.0
                logger.info(
                    f"  [TwoStage Label {c}] Stage2 候選池 FP 混入率: "
                    f"{n_fp_in_pool}/{len(true_c)} ({fp_ratio:.1%})  "
                    f"TP={n_tp_in_pool}, FP={n_fp_in_pool}"
                )

            target_list      = getattr(args, 'targert_list', [])
            save_dir_stage2  = getattr(args, 'Resutl_dir', 'gmm_debug_plots') + '/twostage_plots'
            label_index_path = getattr(args, 'label_index_path', None)
            enc_name         = getattr(args, 'encoder_name', '')

            try:
                mode = self.feature_mode  # '2d' | 'cd' | 'gap'

                if mode == '2d':
                    # ── 原始 2D (−CD, Gap) GMM ──────────────────────────────
                    if self.n_components == 2:
                        prob_clean = self._fit_2d_gmm(cd_c, gap_c)
                        new_labels = self._apply_epsilon_band(prob_clean, epsilon)
                        if c in target_list:
                            visualize_2d_gmm_candidates(
                                cd_vals=cd_c, gap_vals=gap_c, prob_clean=prob_clean,
                                label_index=c, save_dir=save_dir_stage2,
                                label_index_path=label_index_path,
                                encoder_name=enc_name, true_labels=true_c,
                            )
                    else:
                        prob_dict  = self._run_2d_gmm(cd_c, gap_c)
                        new_labels = self._apply_2d_correction_logic(prob_dict, epsilon)

                    # 群數純度分析（k=2..5）：target_list 且有 true_c 時觸發
                    if c in target_list and true_c is not None:
                        visualize_gmm_cluster_purity_k(
                            cd_vals=cd_c, gap_vals=gap_c, true_labels=true_c,
                            label_index=c, k_values=(2, 3, 4, 5),
                            save_dir=save_dir_stage2,
                            label_index_path=label_index_path,
                            encoder_name=enc_name,
                        )

                elif mode in ('cd', 'gap'):
                    # ── 1D GMM：只用單一特徵 ────────────────────────────────
                    # 'cd'  → -CD（翻轉後大值 = FP 嫌疑高）
                    # 'gap' → Positive Gap（大值 = FP 嫌疑高）
                    scores_1d  = -cd_c if mode == 'cd' else gap_c
                    prob_dict  = self._run_1d_gmm(scores_1d)
                    new_labels = self._apply_2d_correction_logic(prob_dict, epsilon)

                    if c in target_list and true_c is not None:
                        visualize_1d_gmm_features(
                            cd_vals=cd_c, gap_vals=gap_c, true_labels=true_c,
                            label_index=c, n_components=self.n_components,
                            save_dir=save_dir_stage2,
                            label_index_path=label_index_path,
                            encoder_name=f"{enc_name}_{mode}",
                        )

                else:
                    raise ValueError(
                        f"Unknown feature_mode='{mode}'. Choose '2d', 'cd', or 'gap'."
                    )

            except Exception as e:
                logger.warning(
                    f"  [TwoStage Label {c}] Stage2 GMM"
                    f"(n={self.n_components}, mode={self.feature_mode}) failed: {e}, skipping."
                )
                continue

            corrected[rest_indices, c] = new_labels

            soft = int(np.sum((new_labels > 0.0) & (new_labels < 1.0)))
            gmm_flipped = int(np.sum(new_labels == 0.0))
            total_soft += soft
            total_flipped += gmm_flipped

            logger.info(
                f"  [TwoStage Label {c}] Stage2 candidates={len(rest_indices)}, "
                f"flipped={gmm_flipped}, soft={soft}"
            )

        logger.info(
            f"[TwoStageRELFPCorrector] Done. "
            f"stage2_candidates={total_candidates}, flipped={total_flipped}, soft={total_soft}"
        )
        return corrected


class FrequencyAware1DGMMCorrector:
    """
    頻率感知 1D GMM 校正器（FP 導向）

    頻率組（head/middle/tail）作為基準，但融合策略由資料驅動決定：
        先計算每個標籤正樣本的 CD–Gap Pearson 相關係數，依此選策略：
            corr > CORR_HIGH (0.6)           → mean  (兩信號一致，平均降噪)
            corr < CORR_LOW  (0.3)           → max   (兩信號不一致，取保守上界)
            pos_count < MIN_CD_SAMPLES (100) → gap   (CD 原型估計不穩)
            其餘依頻率組預設                  → head=max, middle=mean, tail=gap

    主要方法：
        analyze()             — 輸出每個標籤的 CD/Gap 統計 + 自動建議策略（DataFrame）
        correct()             — 一次性依資料驅動 epsilon-band 校正所有正樣本
        progressive_correct() — 按可疑分數降序逐步翻轉，每輪回報 FP Recall
    """

    HEAD_THRESH    = 4500
    MID_THRESH     = 2000
    CORR_HIGH      = 0.6   # above → mean (signals agree)
    CORR_LOW       = 0.3   # below → max  (signals diverge)
    MIN_CD_SAMPLES = 100   # below → gap  (CD prototype unreliable)

    def __init__(self,
                 n_components: int   = 2,
                 head_epsilon: float = 0.03,
                 mid_epsilon:  float = 0.05,
                 tail_epsilon: float = 0.10):
        self.n_components = n_components
        self.head_epsilon = head_epsilon
        self.mid_epsilon  = mid_epsilon
        self.tail_epsilon = tail_epsilon

    # ── 1. 標籤頻率分組 ─────────────────────────────────────────────

    def _classify_labels(self, labels: np.ndarray) -> tuple:
        """
        Returns:
            freq_group : dict  {c → 'head' | 'middle' | 'tail'}
            pos_counts : [C]   每個標籤的正樣本絕對數
        """
        pos_counts = labels.sum(axis=0).astype(int)
        freq_group = {}
        for c, cnt in enumerate(pos_counts):
            if cnt >= self.HEAD_THRESH:
                freq_group[c] = 'head'
            elif cnt >= self.MID_THRESH:
                freq_group[c] = 'middle'
            else:
                freq_group[c] = 'tail'
        return freq_group, pos_counts

    # ── 2. 1D GMM → noisy 後驗概率 ──────────────────────────────────

    def _1d_noisy_prob(self, scores_1d: np.ndarray) -> np.ndarray:
        """
        Mean 最大的 component = noisy（FP 嫌疑最高），回傳其後驗概率。
        若樣本數 < 4 回傳全零。
        """
        n = len(scores_1d)
        if n < 4:
            return np.zeros(n)
        try:
            gmm = GMM(n_components=self.n_components, max_iter=200,
                      tol=1e-3, reg_covar=1e-4, random_state=0)
            gmm.fit(scores_1d.reshape(-1, 1))
        except Exception:
            return np.zeros(n)
        noisy_idx = int(np.argmax(gmm.means_.flatten()))
        return gmm.predict_proba(scores_1d.reshape(-1, 1))[:, noisy_idx]

    # ── 3. 資料驅動策略選擇 ─────────────────────────────────────────

    def _select_strategy(self, group: str, corr: float, pos_count: int) -> str:
        """
        依 CD–Gap Pearson 相關係數 + 正樣本數決定融合策略。
        回傳 'max' | 'mean' | 'gap'
        """
        if pos_count < self.MIN_CD_SAMPLES or group == 'tail':
            return 'gap'
        if corr > self.CORR_HIGH:
            return 'mean'
        if corr < self.CORR_LOW:
            return 'max'
        return 'max' if group == 'head' else 'mean'

    # ── 4. 分布分析 + 策略報告 ──────────────────────────────────────

    def analyze(self,
                cd_scores:  np.ndarray,
                gap_scores: np.ndarray,
                labels:     np.ndarray,
                save_path:  str = None) -> 'pd.DataFrame':
        """
        對每個標籤計算正樣本的 CD/Gap 分布統計與 Pearson 相關係數，
        輸出自動選擇的融合策略供人工審閱。

        Args:
            save_path: 若傳入，將 DataFrame 存為 CSV（e.g. 'result_dir/fa1dgmm_analyze.csv'）

        Returns:
            DataFrame columns:
                label, freq_group, pos_count,
                cd_mean, cd_std, gap_mean, gap_std,
                cd_gap_corr, strategy
        """
        import pandas as pd
        freq_group, _ = self._classify_labels(labels)
        records = []
        for c in range(labels.shape[1]):
            pos_idx = np.where(labels[:, c] == 1)[0]
            n = len(pos_idx)
            if n < 4:
                records.append({
                    'label': c, 'freq_group': freq_group[c], 'pos_count': n,
                    'cd_mean': np.nan, 'cd_std': np.nan,
                    'gap_mean': np.nan, 'gap_std': np.nan,
                    'cd_gap_corr': np.nan, 'strategy': 'skip',
                })
                continue
            cd_c  = cd_scores[pos_idx, c]
            gap_c = gap_scores[pos_idx, c]
            corr  = float(np.corrcoef(cd_c, gap_c)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
            strategy = self._select_strategy(freq_group[c], corr, n)
            records.append({
                'label':       c,
                'freq_group':  freq_group[c],
                'pos_count':   n,
                'cd_mean':     round(float(cd_c.mean()), 4),
                'cd_std':      round(float(cd_c.std()),  4),
                'gap_mean':    round(float(gap_c.mean()), 4),
                'gap_std':     round(float(gap_c.std()),  4),
                'cd_gap_corr': round(corr, 4),
                'strategy':    strategy,
            })
        df = pd.DataFrame(records)
        logger.info(
            "[FA1DGMM.analyze] strategy distribution:\n"
            + df.groupby('strategy')['label'].count().to_string()
        )
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"[FA1DGMM.analyze] saved → {save_path}")
        return df

    # ── 5. 計算可疑分數矩陣（資料驅動策略）────────────────────────

    def compute_suspicion(self,
                          cd_scores:  np.ndarray,
                          gap_scores: np.ndarray,
                          labels:     np.ndarray) -> np.ndarray:
        """
        Returns:
            suspicion : [N, C]，僅 label=1 的位置有值，其餘為 0
        """
        N, C = labels.shape
        freq_group, _ = self._classify_labels(labels)
        suspicion = np.zeros((N, C), dtype=float)

        for c in range(C):
            pos_idx = np.where(labels[:, c] == 1)[0]
            n = len(pos_idx)
            if n < 4:
                continue

            cd_c  = cd_scores[pos_idx, c]
            gap_c = gap_scores[pos_idx, c]
            corr  = float(np.corrcoef(cd_c, gap_c)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0

            strategy  = self._select_strategy(freq_group[c], corr, n)
            cd_noisy  = self._1d_noisy_prob(cd_c)
            gap_noisy = self._1d_noisy_prob(gap_c)

            if strategy == 'max':
                combined = np.maximum(cd_noisy, gap_noisy)
            elif strategy == 'mean':
                combined = (cd_noisy + gap_noisy) / 2.0
            else:  # 'gap'
                combined = gap_noisy

            logger.debug(
                f"  [FA1DGMM c={c:>2} | {freq_group[c]:6} | corr={corr:+.3f}] "
                f"strategy={strategy}"
            )
            suspicion[pos_idx, c] = combined

        return suspicion

    # ── 6. Epsilon-band（noisy prob 方向）───────────────────────────

    def _apply_epsilon_band(self, prob_noisy: np.ndarray, epsilon: float) -> np.ndarray:
        """
        prob_noisy > 0.5 + ε → 0.0  (確定 FP，翻轉)
        prob_noisy < 0.5 - ε → 1.0  (確定 TP，保留)
        中間帶               → 1 - prob_noisy  (soft label)
        """
        p = np.nan_to_num(prob_noisy, nan=0.5)
        return np.where(p > 0.5 + epsilon, 0.0,
               np.where(p < 0.5 - epsilon, 1.0,
                        1.0 - p))

    # ── 7. 一次性全局校正 ────────────────────────────────────────────

    def correct(self,
                cd_scores:  np.ndarray,
                gap_scores: np.ndarray,
                labels:     np.ndarray) -> np.ndarray:
        """
        依資料驅動 epsilon-band 直接校正所有正樣本。
        Returns: corrected [N, C] float
        """
        freq_group, _ = self._classify_labels(labels)
        suspicion     = self.compute_suspicion(cd_scores, gap_scores, labels)
        corrected     = labels.astype(float).copy()
        eps_map       = {'head': self.head_epsilon,
                         'middle': self.mid_epsilon,
                         'tail':   self.tail_epsilon}

        for c in range(labels.shape[1]):
            pos_idx = np.where(labels[:, c] == 1)[0]
            if len(pos_idx) < 4:
                continue
            eps     = eps_map[freq_group[c]]
            new_lab = self._apply_epsilon_band(suspicion[pos_idx, c], eps)
            corrected[pos_idx, c] = new_lab
            flipped = int((new_lab == 0.0).sum())
            soft    = int(((new_lab > 0.0) & (new_lab < 1.0)).sum())
            logger.info(
                f"  [FA1DGMM c={c:>2} | {freq_group[c]:6}] "
                f"pos={len(pos_idx)}, flipped={flipped}, soft={soft}, eps={eps}"
            )

        return corrected

    # ── 6. 逐步校正 + FP Recall 追蹤 ───────────────────────────────

    def progressive_correct(self,
                            cd_scores:  np.ndarray,
                            gap_scores: np.ndarray,
                            labels:     np.ndarray,
                            y_true:     np.ndarray,
                            n_rounds:   int = 10) -> tuple:
        """
        按可疑分數降序逐步翻轉，每輪評估 FP Recall。

        Args:
            cd_scores, gap_scores : [N, C] 已歸一化分數（masked sign convention）
            labels   : [N, C] noisy 標籤 (0/1)
            y_true   : [N, C] 真實標籤 (0/1)
            n_rounds : 分幾輪校正

        Returns:
            corrected   : [N, C] float，最終校正後標籤
            round_stats : list[dict]  每輪統計
        """
        suspicion = self.compute_suspicion(cd_scores, gap_scores, labels)

        pos_rows, pos_cols = np.where(labels == 1)
        sorted_order = np.argsort(-suspicion[pos_rows, pos_cols])
        s_rows = pos_rows[sorted_order]
        s_cols = pos_cols[sorted_order]

        n_pos    = len(s_rows)
        batch_sz = max(1, n_pos // n_rounds)

        corrected    = labels.astype(float).copy()
        fp_mask      = (y_true == 0) & (labels == 1)
        total_fp     = int(fp_mask.sum())
        round_stats  = []

        for r in range(n_rounds):
            start = r * batch_sz
            end   = (r + 1) * batch_sz if r < n_rounds - 1 else n_pos
            corrected[s_rows[start:end], s_cols[start:end]] = 0.0

            corrected_bin   = (corrected >= 0.5).astype(int)
            fix_fp          = int(((corrected_bin == 0) & fp_mask).sum())
            miss_fp         = int(((corrected_bin == 1) & fp_mask).sum())
            fp_recall       = fix_fp / total_fp if total_fp > 0 else 0.0

            flipped_mask                         = np.zeros_like(corrected, dtype=bool)
            flipped_mask[s_rows[:end], s_cols[:end]] = True
            fp_precision = int((flipped_mask & fp_mask).sum()) / end if end > 0 else 0.0

            round_stats.append({
                'round':              r + 1,
                'cumulative_flipped': end,
                'flip_ratio':         end / n_pos,
                'fix_fp':             fix_fp,
                'miss_fp':            miss_fp,
                'total_fp':           total_fp,
                'fp_recall':          fp_recall,
                'fp_precision':       fp_precision,
            })

            logger.info(
                f"  [Progressive {r+1:>2}/{n_rounds}] "
                f"flipped={end}/{n_pos} ({end/n_pos:.1%})  "
                f"Fix_FP={fix_fp}  Miss_FP={miss_fp}  "
                f"FP_Recall={fp_recall:.3f}  FP_Prec={fp_precision:.3f}"
            )

        return corrected, round_stats


class GMMNoiseFilter(BaseNoiseFilter):
    """使用 GMM 二分法進行篩選"""
    def __init__(self, n_components=2, threshold=0.5):
        self.n_components = n_components
        self.threshold = threshold

    def filter(self, scores, indices):
        scores = np.asarray(scores, dtype=float)
        indices = np.asarray(indices)
        
        # --- 形狀診斷 ---
        N_indices = len(indices)
        N_scores = len(scores)
        
        print(f"[{self.__class__.__name__}] Input Check -> Scores: {scores.shape}, Indices: {indices.shape}")
        print(f"  - N_indices: {N_indices}, N_scores: {N_scores}")
        if scores.ndim == 2 and scores.shape[0] == N_indices:
            print(f"  > 偵測到矩陣分數，正在聚合 (Sum axis=1)...")
            scores = scores.sum(axis=1)

        # 情況 B: Scores 是 [N] 且長度一致 -> 正常情況
        elif scores.ndim == 1 and N_scores == N_indices:
            pass # 這裡不用做任何事
            
        # 3. 處理 Flatten 後長度不匹配的情況 (防呆)
        # 如果 scores 已經被攤平了 (1D)，但長度是 indices 的倍數 (例如 181710 vs 53840)
        if scores.ndim == 1 and len(scores) != len(indices):
            if len(scores) > len(indices) and len(scores) % len(indices) == 0:
                num_classes = len(scores) // len(indices)
                print(f"[{self.__class__.__name__}] 偵測到攤平的 Multi-label 輸入。正在 Reshape 並聚合...")
                scores = scores.reshape(len(indices), num_classes).sum(axis=1)
            else:
                # 如果無法整除，代表資料真的對不起來，必須報錯
                raise ValueError(f"維度嚴重錯誤！Scores: {len(scores)}, Indices: {len(indices)}。無法對齊。")
            
        X = scores.reshape(-1, 1)
        
        # Fit GMM
        gmm = GMM(n_components=self.n_components, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(X)
        
        # 找出 Mean 較小的那群 (假設 Loss/Distance 小的是乾淨的) 
        clean_idx = gmm.means_.argmin()
        probs = gmm.predict_proba(X)
        prob_clean = probs[:, clean_idx]
        
        # Thresholding
        is_clean = prob_clean > self.threshold
        
        clean_indices = indices[is_clean]
        noisy_indices = indices[~is_clean]
        
        print(f"[{self.__class__.__name__}] Filter Report:")
        # print(f"  - Total: {len(indices)}")
        # print(f"  - Clean: {len(clean_indices)} ({len(clean_indices)/len(indices):.2%})")
        # print(f"  - Noisy: {len(noisy_indices)} ({len(noisy_indices)/len(indices):.2%})")
        
        return clean_indices, noisy_indices
    
    def _run_gmm_on_subset(self, scores_subset):
        """
        內部 helper：在子集上執行 GMM 並回傳「屬於乾淨群(低分群)的機率」
        """
        if len(scores_subset) <= 1:
            return None
            
        data = scores_subset.reshape(-1, 1)
        gmm = GMM(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4, random_state=0)
        gmm.fit(data)
        
        # 找出 Mean 較小的那群 (代表 Low Loss / Clean)
        clean_comp_idx = gmm.means_.argmin()
        probs_all = gmm.predict_proba(data)
        
        # 回傳 "屬於乾淨群" 的機率
        return probs_all[:, clean_comp_idx]
    
    
        
    def _apply_band_correction(self, probs,y, args):
        """
        內部 helper：套用 epsilon band 進行數值校正
        probs: 屬於 '乾淨/正類' 的機率值 (1D array)
        """
        band = (0.5 - args.epsilon, 0.5 + args.epsilon)
        
        # 統計落在模糊地帶的數量 (僅供 Debug)
        n_mid = np.sum((probs >= band[0]) & (probs <= band[1]))
        if n_mid > 0:
            print(f'  [Correction] Uncertain samples (in band): {n_mid} / {len(probs)}')

        # 處理 NaN，預設為 0.5 (不確定)
        probs = np.nan_to_num(probs, nan=0.5)
        

        # 中間 -> 保持原始機率 (Soft Label)
        out = np.where(probs > 0.5 + args.epsilon, y,
              np.where(probs < 0.5 - args.epsilon, 1-y, probs))
        
        return out.astype(float)

    def correction(self, scores, labels, args):
        """
        進階校正：針對正樣本進行 GMM 重算並修正標籤
        
        Args:
            scores: [N] 所有的分數 (HSM score 或 Loss)
            labels: [N] 原始標籤 (0 或 1)
            args: 包含 args.epsilon 的參數物件
        Returns:
            corrected_labels: [N] 校正後的標籤 (浮點數，包含 0.0, 1.0 或中間值)
        """
        print(f"[{self.__class__.__name__}] Running Correction...")
        
        # 1. 資料前處理 資料攤平
        scores_flat = np.asarray(scores, dtype=float).ravel()
        labels_flat = np.asarray(labels, dtype=int).ravel()
        
        # 🔍 檢查點：現在這裡絕對不會報錯了
        if len(scores_flat) != len(labels_flat):
            raise ValueError(f"Shape mismatch! Scores: {len(scores_flat)}, Labels: {len(labels_flat)}")

        scores_flat = np.nan_to_num(scores_flat, nan=1.0, posinf=1.0, neginf=0.0)

        # 2. 接下來的邏輯跟之前一樣
        corrected_labels_flat = labels_flat.astype(float).copy()
        pos_f = 1
        neg_f = 0
        pos_mask = (labels_flat == pos_f) # 這時候長度一致，Mask 運作正常
        neg_mask = (labels_flat == neg_f)

        # === A. 正樣本處理 ===
        pos_scores = scores_flat[pos_mask] 
        print(f"  [+] Processing {len(pos_scores)} positive labels...")
        
        if len(pos_scores) > 1:

            pro_b_clean_pos = self._run_gmm_on_subset(pos_scores)
            if pro_b_clean_pos is not None:
                new_pos_labels = self._apply_band_correction(pro_b_clean_pos ,pos_f, args)
                corrected_labels_flat[pos_mask] = new_pos_labels
                

        # === B. 負樣本處理 ===
        neg_scores = scores_flat[neg_mask]
        print(f"  [-] Processing {len(neg_scores)} negative labels...")
        
        if len(neg_scores) > 1:
            pro_b_clean_neg = self._run_gmm_on_subset(neg_scores)
            if pro_b_clean_neg is not None:
                new_neg_labels = self._apply_band_correction(pro_b_clean_neg,neg_f, args)
                corrected_labels_flat[neg_mask] = new_neg_labels

        # 3. 最後看你要回傳攤平的結果，還是 Reshape 回去
        # 如果 Dataset 預期的是攤平的 Label，就直接回傳
        # 如果 Dataset 預期的是 [N, C]，請做 reshape
        # return corrected_labels_flat.reshape(labels.shape) 
        corrected_labels = corrected_labels_flat.reshape(labels.shape)
        for label in range(labels.shape[1]):
            num_pos = np.sum(labels[:, label] == 1)
            num_neg = np.sum(labels[:, label] == 0)
            num_pos_corrected = np.sum(corrected_labels[:, label] == 1)
            num_neg_corrected = np.sum(corrected_labels[:, label] == 0)
            # logger.info(f"  [Label {label} Correction Report]")
            # logger.info(f"    - Original Positives: {num_pos}, Negatives: {num_neg}")
            # 訂正結果 origin -flip
            pos_flip_neg= np.sum((labels[:, label] == 1) & (corrected_labels[:, label] == 0))
            neg_flip_pos= np.sum((labels[:, label] == 0) & (corrected_labels[:, label] == 1))
            
            logger.info("    - Total Changes:")
            logger.info(f"    - Positives flipped to Negative (1->0): {pos_flip_neg}")
            logger.info(f"    - Negatives flipped to Positive (0->1): {neg_flip_pos}")
            logger.info(f"    - After Correction: Positives: {num_pos_corrected}, Negatives: {num_neg_corrected}")
            print(f"  > 完成 {label} 個類別的獨立校正。")
        return corrected_labels
    

    def correction_perlabel(self, scores, labels, args):
        # 依 args.Noise_type 決定要修哪些子集：
        #   'FP' → 只修 y=1 子集（雜訊只在 positives 上，y=0 不該動）
        #   'FN' → 只修 y=0 子集
        #   'ALL'（或未設定）→ 兩邊都修，維持舊行為
        noise_type = getattr(args, 'Noise_type', 'ALL')
        correct_pos = noise_type in ('FP', 'ALL')
        correct_neg = noise_type in ('FN', 'ALL')
        print(f"[{self.__class__.__name__}] Running Per-Label GMM Correction... "
              f"noise_type={noise_type}  correct_pos={correct_pos}  correct_neg={correct_neg}")

        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        scores = np.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)

        N, num_classes = scores.shape

        # 建立輸出的矩陣 (複製一份)
        refined_labels = labels.astype(float).copy()

        # 用來統計進度的 bar
        iterator = range(num_classes)
        # 如果類別很多，可以考慮用 tqdm 包起來: tqdm(range(num_classes), desc="Label Correction")

        # 2. 針對每一個類別 (Column) 獨立處理
        for c in iterator:


            stats = {
            "pos_origin": 0,
            "neg_origin": 0,
            "pos_flipped_to_neg": 0, # 1 -> 0 (找出假正例)
            "neg_flipped_to_pos": 0, # 0 -> 1 (找出假負例)
            "soft_labels": 0,        # 變成 0.x (不確定)

        }
            # 取出第 c 個類別的所有樣本數據
            col_scores = scores[:, c]  # [N]
            col_labels = labels[:, c]  # [N]

            # 定義 Mask
            pos_f = 1
            neg_f = 0

            pos_mask = (col_labels == pos_f)
            neg_mask = (col_labels == neg_f)

            # -------------------------------------------
            # (A) 該類別的正樣本 (Positive) 校正
            # -------------------------------------------
            pos_scores_c = col_scores[pos_mask]
            stats['pos_origin'] = len(pos_scores_c)

            # 只有當該類別的正樣本夠多時，才跑 GMM (避免 sample 太少 GMM 炸開)
            target_watch_list = args.targert_list  # 你可以指定一些類別來畫圖檢查
            if correct_pos and len(pos_scores_c) > 1:
                prob_clean = self._run_gmm_on_subset(pos_scores_c)
                if c in target_watch_list:
                    # 顯示結果
                    visualize_gmm(pos_scores_c, class_name=c, subset_type=f"Pos_Original{args.alpha}",save_dir=f"gmm_debug_plots{args.alpha}", label_index_path=getattr(args, 'label_index_path', 'dataset/AAPD/label_to_index.json'))
                if prob_clean is not None:
                    # 2. 傳入機率，Target=1.0
                    # Prob高(乾淨) -> 維持 1
                    # Prob低(雜訊) -> 翻轉為 0
                    new_pos = self._apply_band_correction(prob_clean, 1.0, args)
                    refined_labels[pos_mask, c] = new_pos
                    # --- 統計變化 ---
                    # 原本是 1，變成了 0 (完全翻轉)
                    flipped_0 = np.sum(new_pos == 0.0)
                    # 變成了軟標籤 (0 < x < 1)
                    soft = np.sum((new_pos > 0.0) & (new_pos < 1.0))
                    stats["pos_flipped_to_neg"] += flipped_0
                    stats["soft_labels"] += soft
            # -------------------------------------------
            # (B) 該類別的負樣本 (Negative) 校正
            neg_scores_c = col_scores[neg_mask]
            stats['neg_origin'] = len(neg_scores_c)

            if correct_neg and len(neg_scores_c) > 1:
                prob_clean = self._run_gmm_on_subset(neg_scores_c)
                if c in target_watch_list:

                    visualize_gmm(neg_scores_c, class_name=c, subset_type=f"Neg_Original{args.alpha}",save_dir=f"gmm_debug_plots{args.alpha}", label_index_path=getattr(args, 'label_index_path', 'dataset/AAPD/label_to_index.json'),encoder_name=args.encoder_name)
                if prob_clean is not None:
                    # 2. 傳入機率，Target=0.0
                    # Prob高(乾淨) -> 維持 0
                    # Prob低(雜訊) -> 翻轉為 1
                    new_neg = self._apply_band_correction(prob_clean, 0.0, args)
                    refined_labels[neg_mask, c] = new_neg
                    # --統計
                    flipped_1 = np.sum(new_neg == 1.0)
                    # 變成了軟標籤
                    soft = np.sum((new_neg > 0.0) & (new_neg < 1.0))
                    stats["neg_flipped_to_pos"] += flipped_1
                    stats["soft_labels"] += soft
            # --- 每個類別的校正報告 ---
            logger.info(f"  [Label {c} Correction Report]")
            logger.info(f"    - Original Positives: {stats['pos_origin']}, Negatives: {stats['neg_origin']}")
            # 訂正結果 origin -flip
            pos_change=stats['pos_origin']- stats['pos_flipped_to_neg']+stats['neg_flipped_to_pos']
            neg_change=stats['neg_origin']- stats['neg_flipped_to_pos']+stats['pos_flipped_to_neg']
            logger.info("    - Total Changes:")
            logger.info(f"    - Positives flipped to Negative (1->0): {stats['pos_flipped_to_neg']}")
            logger.info(f"    - Negatives flipped to Positive (0->1): {stats['neg_flipped_to_pos']}")
            logger.info(f"    - Soft Labels assigned: {stats['soft_labels']}")
            logger.info(f"    - After Correction: Positives: {pos_change}, Negatives: {neg_change}")

        print(f"  > 完成 {num_classes} 個類別的獨立校正。")
        
        return refined_labels
    
class LabelRefiner:
    def __init__(self, n_components=2, random_state=0):
        self.n_components = n_components
        self.random_state = random_state
        
        
    def _run_gmm(self, scores_subset):
        """執行 GMM 並回傳各樣本屬於不同成分的機率"""
        if len(scores_subset) <= self.n_components:
            return None
            
        data = scores_subset.reshape(-1, 1)
        gmm = GMM(
            n_components=self.n_components, 
            max_iter=100, 
            tol=1e-2, 
            reg_covar=5e-4, 
            random_state=self.random_state
        )
        gmm.fit(data)
        
        # 排序 Means：索引 0 永遠是 Low Loss (Clean)
        sorted_indices = np.argsort(gmm.means_.flatten())
        probs_all = gmm.predict_proba(data)
        
        # 回傳字典，方便後續擴張
        result = {
            'clean_probs': probs_all[:, sorted_indices[0]],
            'noisy_probs': probs_all[:, sorted_indices[-1]], # 最後一個永遠是最高 Loss
        }
        # 通用處理中間群 (支援 3, 4, 5+ components)
        mid_indices = sorted_indices[1:-1]
        if len(mid_indices) == 1:
            # 3-comp: 保持向後相容，使用 'mid_probs' key
            result['mid_probs'] = probs_all[:, mid_indices[0]]
        else:
            # 4, 5+ comp: 使用 'mid_probs_0', 'mid_probs_1', ...
            for i, mid_idx in enumerate(mid_indices):
                result[f'mid_probs_{i}'] = probs_all[:, mid_idx]
            
        return result

    def _apply_correction_logic(self, prob_dict, target_y, args):
        """
        核心校正邏輯：
        2-comp: 使用 epsilon band
        3-comp: 使用最大機率所屬群集
        4-comp: 漸進式 soft label (clean → mid-clean → mid-noisy → noisy)
        5-comp: 漸進式 soft label (clean → mid-clean → uncertain → mid-noisy → noisy)
        """
        if self.n_components == 2:
            # 原本的 Epsilon 邏輯
            probs = prob_dict['clean_probs']
            probs = np.nan_to_num(probs, nan=0.5)
            
            # y=1 時: Clean(1), Noisy(0), Band(prob)
            # y=0 時: Clean(0), Noisy(1), Band(1-prob)
            out = np.where(probs > 0.5 + args.epsilon, target_y,
                  np.where(probs < 0.5 - args.epsilon, 1.0 - target_y, 
                  probs if target_y == 1.0 else 1.0 - probs))
            return out
            
        elif self.n_components == 3:
            # 3-comp 自動化邏輯
            p_clean = prob_dict['clean_probs']
            p_mid = prob_dict['mid_probs']
            p_noisy = prob_dict['noisy_probs']
            
            stack = np.vstack([p_clean, p_mid, p_noisy])
            winners = np.argmax(stack, axis=0)
            
            out = np.zeros_like(p_clean)
            out[winners == 0] = target_y      # 歸類為乾淨
            out[winners == 2] = 1.0 - target_y # 歸類為雜訊
            out[winners == 1] = 0.5            # 歸類為模糊
            return out

        elif self.n_components == 4:
            # 4-comp 漸進式邏輯：clean / mid-clean / mid-noisy / noisy
            p_clean = prob_dict['clean_probs']
            p_mid_clean = prob_dict['mid_probs_0']
            p_mid_noisy = prob_dict['mid_probs_1']
            p_noisy = prob_dict['noisy_probs']

            stack = np.vstack([p_clean, p_mid_clean, p_mid_noisy, p_noisy])
            winners = np.argmax(stack, axis=0)

            # 漸進 soft label: target_y → 0.7 → 0.3 → 1-target_y
            gradient = [target_y, 0.7 if target_y == 1.0 else 0.3,
                        0.3 if target_y == 1.0 else 0.7, 1.0 - target_y]
            out = np.zeros_like(p_clean)
            for i, val in enumerate(gradient):
                out[winners == i] = val
            return out

        elif self.n_components == 5:
            # 5-comp 漸進式邏輯：clean / mid-clean / uncertain / mid-noisy / noisy
            p_clean = prob_dict['clean_probs']
            p_mid_clean = prob_dict['mid_probs_0']
            p_uncertain = prob_dict['mid_probs_1']
            p_mid_noisy = prob_dict['mid_probs_2']
            p_noisy = prob_dict['noisy_probs']

            stack = np.vstack([p_clean, p_mid_clean, p_uncertain, p_mid_noisy, p_noisy])
            winners = np.argmax(stack, axis=0)

            # 漸進 soft label: target_y → 0.75 → 0.5 → 0.25 → 1-target_y
            gradient = [target_y, 0.75 if target_y == 1.0 else 0.25,
                        0.5,
                        0.25 if target_y == 1.0 else 0.75, 1.0 - target_y]
            out = np.zeros_like(p_clean)
            for i, val in enumerate(gradient):
                out[winners == i] = val
            return out

    def refine(self, scores, labels, args):
        """
        對多標籤/多類別進行校正
        scores: [N, num_classes]
        labels: [N, num_classes]
        """
        logger.info(f"Starting {self.n_components}-component GMM Correction...")
        
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=1.0, posinf=1.0, neginf=0.0)
        labels = np.asarray(labels, dtype=int)
        refined_labels = labels.astype(float).copy()
        
        N, num_classes = scores.shape
        
        # 檢查是否有指定要繪圖的標籤列表
        target_watch_list = getattr(args, 'targert_list', [])
        save_dir = f"gmm_debug_plots{getattr(args, 'alpha', '')}"

        for c in range(num_classes):
            col_scores = scores[:, c]
            col_labels = labels[:, c]
            
            # 建立該類別統計
            stats = {"1->0": 0, "0->1": 0, "soft": 0, "pos_orig": 0, "neg_orig": 0}

            for target_y in [1, 0]:
                mask = (col_labels == target_y)
                subset_scores = col_scores[mask]
                stats["pos_orig" if target_y == 1 else "neg_orig"] = len(subset_scores)

                if len(subset_scores) > self.n_components:
                    prob_dict = self._run_gmm(subset_scores)
                    
                    # 如果該標籤在觀察列表中,則繪製 GMM 圖
                    if c in target_watch_list and prob_dict is not None:
                        subset_type = f"{'Pos' if target_y == 1 else 'Neg'}_Original{getattr(args, 'alpha', '')}"
                        visualize_gmm(
                            subset_scores,
                            class_name=c,
                            subset_type=subset_type,
                            save_dir=save_dir,
                            n_components=self.n_components,
                            label_index_path=getattr(args, 'label_index_path', 'dataset/AAPD/label_to_index.json')
                        )
                    
                    if prob_dict is not None:
                        new_vals = self._apply_correction_logic(prob_dict, float(target_y), args)
                        refined_labels[mask, c] = new_vals
                        
                        # 統計
                        if target_y == 1:
                            stats["1->0"] = np.sum(new_vals == 0.0)
                        else:
                            stats["0->1"] = np.sum(new_vals == 1.0)
                        stats["soft"] += np.sum((new_vals > 0.0) & (new_vals < 1.0))

            self._log_report(c, stats)

        return refined_labels

    def _log_report(self, c, stats):
        logger.info(f" Label {c} | Orig P/N: {stats['pos_orig']}/{stats['neg_orig']} | "
                         f"Flipped: 1->0:{stats['1->0']}, 0->1:{stats['0->1']} | Soft: {stats['soft']}")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation building blocks (per-cell BCE, Stage1 flipper, GMM(3) intersection)
# ─────────────────────────────────────────────────────────────────────────────

class StandardLossPerCellCalculator(BaseScoreCalculator):
    """
    Per-cell BCE loss [N, C]，drop-in for HSMHybridPipeline 的 rel_calculator slot。
    跟 RankWeightedLossCalculator 一致：回傳 (indices, scores[N,C], labels[N,C])。
    用於 HSM (a) BCE-only ablation。
    """
    def calculate(self, model, dataloader, device):
        model.eval()
        scores, indices, labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Per-cell BCE Loss"):
                b_ids   = batch['input_ids'].to(device)
                b_mask  = batch['attention_mask'].to(device)
                b_lbl   = batch['labels'].to(device)
                b_idx   = batch['index']

                logits, _ = model(input_ids=b_ids, attention_mask=b_mask)
                loss = F.binary_cross_entropy_with_logits(
                    logits, b_lbl.float(), reduction='none'
                )  # [B, C]

                scores.append(loss.cpu().numpy())
                indices.extend(b_idx.numpy())
                labels.append(b_lbl.cpu().numpy())

        return np.array(indices), np.vstack(scores), np.vstack(labels)


class StageOneRELFlipper:
    """
    Stage 1 (FP only): 對每個 label c，挑 noisy_labels[:, c] == 1 的樣本中
    REL 分數最高的 top_ratio，把 cell 翻成 0。

    與 RELOnlyCorrector.correct() 邏輯一致，但回傳 (flipped_labels, flip_mask)
    讓上層 (TwoStagePipeline) 知道哪些 cell 被翻了。
    """
    def __init__(self, top_ratio: float = 0.01):
        self.top_ratio = top_ratio

    def apply(self, rel_scores: np.ndarray, noisy_labels: np.ndarray):
        rel    = np.asarray(rel_scores, dtype=float)
        labels = np.asarray(noisy_labels, dtype=int)
        N, C   = labels.shape
        flipped   = labels.astype(np.float32).copy()
        flip_mask = np.zeros_like(labels, dtype=bool)

        total = 0
        for c in range(C):
            pos_idx = np.where(labels[:, c] == 1)[0]
            if len(pos_idx) < 4:
                continue
            top_k = max(1, int(len(pos_idx) * self.top_ratio))
            local_top = np.argsort(rel[pos_idx, c])[-top_k:]
            cand = pos_idx[local_top]
            flipped[cand, c]    = 0.0
            flip_mask[cand, c]  = True
            total += len(cand)

        logger.info(f"[StageOneRELFlipper] top_ratio={self.top_ratio} flipped {total} cells "
                    f"({total / max(labels.sum(), 1):.2%} of positive cells)")
        return flipped, flip_mask


class IntersectionGMM3Filter:
    """
    對每個 label c，分別在 cd_scores[:, c] 與 gap_scores[:, c] 上 fit GMM(n=3)，
    把「平均分數最高那群」當作 suspicious cluster (high-suspect)，回傳兩個 mask；
    intersect() 取 AND。

    僅在 noisy_labels[:, c] == 1 的子集上 fit (因為 score 經過 *(2y-1) sign convention，
    label==0 的樣本意義不同)，未被 fit 的 cell 視為 not-suspicious。
    """
    def __init__(self, n_components: int = 3, min_positives: int = 6):
        self.n_components = n_components
        self.min_positives = min_positives

    def _suspicious_mask_one(self, score_col: np.ndarray,
                              label_col: np.ndarray) -> np.ndarray:
        """單一 label：回傳該 label 的 [N] bool mask."""
        N = score_col.shape[0]
        mask = np.zeros(N, dtype=bool)
        pos_idx = np.where(label_col == 1)[0]
        if len(pos_idx) < self.min_positives:
            return mask  # 不足以 fit，全部視為 not suspicious

        x = score_col[pos_idx].reshape(-1, 1)
        # 若 variance 過低 GMM 會失敗，加 reg_covar 緩衝
        try:
            gmm = GMM(n_components=self.n_components, max_iter=200, tol=1e-3,
                      reg_covar=1e-4, random_state=0)
            gmm.fit(x)
        except Exception as e:
            logger.warning(f"[IntersectionGMM3Filter] GMM fit failed: {e}; skip.")
            return mask

        # 平均分數最高的 component → suspicious cluster
        means = gmm.means_.flatten()
        suspect_comp = int(np.argmax(means))
        labels_pred = gmm.predict(x)
        suspect_local = (labels_pred == suspect_comp)
        mask[pos_idx[suspect_local]] = True
        return mask

    def fit_predict_suspicious_mask(self,
                                    score_matrix: np.ndarray,
                                    labels: np.ndarray) -> np.ndarray:
        """回傳 [N, C] bool mask = 該 cell 落在 per-label GMM 最可疑那群。"""
        score_matrix = np.asarray(score_matrix, dtype=float)
        labels       = np.asarray(labels, dtype=int)
        N, C = score_matrix.shape
        out  = np.zeros((N, C), dtype=bool)
        for c in range(C):
            out[:, c] = self._suspicious_mask_one(score_matrix[:, c], labels[:, c])
        return out

    def intersect(self,
                  cd_scores: np.ndarray,
                  gap_scores: np.ndarray,
                  labels: np.ndarray) -> np.ndarray:
        """CD ∩ Gap suspicious masks (AND)."""
        cd_mask  = self.fit_predict_suspicious_mask(cd_scores, labels)
        gap_mask = self.fit_predict_suspicious_mask(gap_scores, labels)
        inter    = cd_mask & gap_mask
        logger.info(f"[IntersectionGMM3Filter] cd_mask={cd_mask.sum()} "
                    f"gap_mask={gap_mask.sum()} intersect={inter.sum()}")
        return inter


class TwoStagePipeline:
    """
    Your method orchestrator:
        Stage 1 (optional): RELTopK flip on per-label top top_ratio (FP only)
        Stage 2 (optional): CD GMM(3) ∩ Gap GMM(3) suspicious cells →
            'flip'    : 翻轉這些 cells
            'discard' : 把這些 cells 從 BCE loss mask 掉 (cell-level)
            'none'    : 不動

    Returns
    -------
    final_labels : [N, C] float32  (Stage1 flipped, Stage2 conditionally flipped)
    loss_mask    : [N, C] float32  (1=參與 BCE loss, 0=丟棄)
    """
    def __init__(self,
                 use_stage1: bool = True,
                 stage2_action: str = 'flip',
                 top_ratio: float = 0.01,
                 n_components: int = 3):
        assert stage2_action in ('flip', 'discard', 'none')
        self.use_stage1    = use_stage1
        self.stage2_action = stage2_action
        self.top_ratio     = top_ratio
        self.n_components  = n_components

    def run(self,
            rel_scores: np.ndarray,
            cd_scores: np.ndarray,
            gap_scores: np.ndarray,
            noisy_labels: np.ndarray):
        labels = np.asarray(noisy_labels, dtype=np.float32).copy()
        N, C   = labels.shape
        loss_mask = np.ones((N, C), dtype=np.float32)

        # ── Stage 1 ──────────────────────────────────────────────
        if self.use_stage1:
            flipper = StageOneRELFlipper(top_ratio=self.top_ratio)
            labels, _ = flipper.apply(rel_scores, labels.astype(int))
            labels = labels.astype(np.float32)

        # ── Stage 2 ──────────────────────────────────────────────
        if self.stage2_action == 'none':
            return labels, loss_mask

        gmm_filter = IntersectionGMM3Filter(n_components=self.n_components)
        # 在「Stage 1 已翻轉後」的 labels 上做 Stage 2 (依舊只看正類 cells)
        susp_mask = gmm_filter.intersect(cd_scores, gap_scores, labels.astype(int))

        if self.stage2_action == 'flip':
            # 翻轉 (對 cell): 1→0 / 0→1
            labels[susp_mask] = 1.0 - labels[susp_mask]
        elif self.stage2_action == 'discard':
            loss_mask[susp_mask] = 0.0

        logger.info(f"[TwoStagePipeline] use_stage1={self.use_stage1} "
                    f"stage2={self.stage2_action} affected={int(susp_mask.sum())} cells")
        return labels, loss_mask
