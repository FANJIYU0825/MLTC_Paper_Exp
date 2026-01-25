
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture as GMM
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from util.logger import logger
from .experence import debug_plot_gmm,visualize_gmm

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
    def minmax_normalize_columnwise(matrix):
        """
        label_wise normalize_array
        
        """
        # 避免修改原始數據
        out = matrix.astype(float, copy=True)
        # 針對每一行 (Column) 找 min 和 max
        min_vals = out.min(axis=0)
        max_vals = out.max(axis=0)
        
        # 避免除以 0 (如果該標籤所有分數都一樣)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-8 
        
        # Broadcasting 運算: (N, C) - (C,) / (C,)
        out = (out - min_vals) / range_vals
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
    def __init__(self, theta=3.0):
        self.theta = theta

    def _calculate_rank_weight(self, logits):
        B, C = logits.shape
        idx_sorted = torch.argsort(logits, dim=1, descending=True)
        ranks = torch.empty_like(idx_sorted, dtype=torch.float, device=logits.device)
        base = torch.arange(1, C + 1, device=logits.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        ranks.scatter_(dim=1, index=idx_sorted, src=base)
        
        w = torch.log(ranks) + 1.0
        theta_tensor = torch.full_like(w, float(self.theta))
        w = torch.minimum(w, theta_tensor)
        return w

    def calculate(self, model, dataloader, device):
        model.eval()
        scores = []
        indices = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Rank Weighted Loss"):
                b_input_ids = batch['input_ids'].to(device)
                b_attn_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                b_idx = batch['index']
           
                logits, _ = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.float(), reduction='none')
                
                weights = self._calculate_rank_weight(logits=logits)

                weighted_loss = loss * weights

                scores.append(weighted_loss.cpu().numpy()) 
                indices.extend(b_idx.numpy())
                labels.append(b_labels.cpu().numpy())
        scores_np = np.vstack(scores)
        indices_np = np.array(indices)
        labels_np = np.vstack(labels)
        print(f"Debug Calculator Output:")
        print(f"  - Scores shape: {scores_np.shape}") # 應該是 (53840, 54)
        print(f"  - Indices shape: {indices_np.shape}") # 應該是 (53840,)
        
        return indices_np, scores_np, labels_np

class PrototypeDistanceCalculator(BaseScoreCalculator):

    """計算 Prototype Distance (CD)"""
    def __init__(self, args):
        self.args = args
        self.prototype_dict = {}

    def _get_threshold(self, logit_list):
        arr = MathUtils.sanitize_array(np.array(logit_list).reshape(-1))
        if arr.size == 0: return 1.0
        mean_logit = arr.mean()
        if mean_logit == 0 or not np.isfinite(mean_logit): return 1.0
        record = [(x / mean_logit) if x > mean_logit else 1.0 for x in arr if np.isfinite(x)]
        return float(np.mean(record)) if len(record) else 1.0

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
                
                B, H = features.shape
                C = self.args.label_size
                rep_np = features.cpu().numpy()
                log_np = logits.cpu().numpy()
                lab_np = labels.cpu().numpy()

                for b in range(B):
                    for c in range(C):
                        if lab_np[b, c] == 1:
                            vec = rep_np[b:b+1, :]
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
            if log_list is None: log_list = np.random.rand(1)
            thr = self._get_threshold(log_list)
            
            # Get single prototype
            if feat_list is None:
                proto = np.zeros(features.shape[1])
            else:
                # 簡化的 prototype 計算
                logs = MathUtils.sanitize_array(np.array(log_list).reshape(-1), -np.inf)
                feats = MathUtils.sanitize_array(np.array(feat_list))
                mask = logs > float(thr)
                if np.any(mask):
                    cand = feats[mask]
                    proto = np.nanmean(cand, axis=0) if cand.size > 0 else np.nanmean(feats, axis=0)
                else:
                    proto = np.nanmean(feats, axis=0)
                
            self.prototype_dict[key] = np.nan_to_num(proto)

    def _compute_vectorized_cd(self, features, device):
        B, H = features.shape
        C = len(self.prototype_dict)
        protos = []
        for i in range(C):
            p = self.prototype_dict.get(i)
            if p is None: p = np.zeros(H)
            protos.append(torch.tensor(p, dtype=features.dtype, device=device))
        
        protos_tensor = torch.stack(protos) # [C, H]
        # features [B, 1, H], protos [1, C, H]
        sim = F.cosine_similarity(features.unsqueeze(1), protos_tensor.unsqueeze(0), dim=2, eps=1e-8)
        # dist = 1.0 - sim
        dist = -1*sim
        
        # Handle empty prototypes (masking)
        has_proto = (protos_tensor.abs().sum(dim=1) > 0).float().unsqueeze(0)
        dist = dist * has_proto + (1.0 - has_proto) * 1.0
        return dist

    def calculate(self, model, dataloader, device):
        # 1. Build Prototypes first
        self._build_prototypes(model, dataloader, device)
        
        # 2. Compute Distances
        model.eval()
        scores = []
        indices = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Prototype Distance"):
                b_input_ids = batch['input_ids'].to(device)
                b_attn_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                b_idx = batch['index']
                
                _, features = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                
                # 計算距離矩陣 [B, C]
                dists = self._compute_vectorized_cd(features, device)
                
                # 這裡使用 Sum 作為該樣本的整體分數 (保持與 REL 一致)
                # dists = torch.clamp(dists, 0.0, 1.0)
                
                
                scores.append(dists.cpu().numpy())
                indices.extend(b_idx.numpy())
                labels.append(b_labels.cpu().numpy())
                

        scores_np = np.vstack(scores)
        labels_np = np.vstack(labels)
        indices_np = np.array(indices)
        return indices_np, scores_np, labels_np
    

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
        print(f"  - Total: {len(indices)}")
        print(f"  - Clean: {len(clean_indices)} ({len(clean_indices)/len(indices):.2%})")
        print(f"  - Noisy: {len(noisy_indices)} ({len(noisy_indices)/len(indices):.2%})")
        
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
        return probs_all[:,0]
    
    def _apply_band_correction_analyzed(probs, y_noisy, args, y_clean):
        """
        probs:    校正階段的預測機率
        y_noisy:  混入雜訊階段的標籤 (Input)
        y_clean:  資料乾淨階段的標籤 (Ground Truth)
        """
        
        # 1. 執行校正邏輯 (取得 y_corrected)
        band = (0.5 - args.epsilon, 0.5 + args.epsilon)
        probs = np.nan_to_num(probs, nan=0.5)
        
        # 生成校正後的標籤 (Soft labels)
        y_corrected_soft = np.where(probs > 0.5 + args.epsilon, 1.0,
                        np.where(probs < 0.5 - args.epsilon, 0.0, 
                        probs))
        
        # 為了進行 0/1 狀態比對，我們將校正結果轉為硬標籤
        # 注意：這裡假設 soft label > 0.5 就是 1，您可以依需求調整
        y_corrected = (y_corrected_soft > 0.5).astype(int)
        
        # 確保輸入都是 numpy int 格式
        if hasattr(y_clean, 'cpu'): y_clean = y_clean.cpu().numpy().astype(int)
        if hasattr(y_noisy, 'cpu'): y_noisy = y_noisy.cpu().numpy().astype(int)
        
        # 2. 計算三階段狀態碼 (Magic Code)
        # Code = Clean(4) + Noisy(2) + Corrected(1)
        # 例如: 101 -> 4 + 0 + 1 = 5
        state_codes = (y_clean << 2) | (y_noisy << 1) | y_corrected
        
        # 3. 統計與分類
        stats = {
            # --- (A) 成功修復 (原本有雜訊 -> 修好了) ---
            '101 (校正對了)': np.sum(state_codes == 5), # Clean=1, Noisy=0, Corr=1
            '010 (校正對了)': np.sum(state_codes == 2), # Clean=0, Noisy=1, Corr=0
            
            # --- (B) 破壞性校正 (原本沒雜訊 -> 改壞了) ---
            '110 (校正錯了)': np.sum(state_codes == 6), # Clean=1, Noisy=1, Corr=0
            '001 (校正錯了)': np.sum(state_codes == 1), # Clean=0, Noisy=0, Corr=1
            
            # --- (C) 無效校正/漏改 (原本有雜訊 -> 沒修到) ---
            '100 (沒校正到)': np.sum(state_codes == 4), # Clean=1, Noisy=0, Corr=0
            '011 (沒校正到)': np.sum(state_codes == 3), # Clean=0, Noisy=1, Corr=1
            
            # --- (D) 正確保持 (原本沒雜訊 -> 保持原樣) ---
            '111 (不需要校正)': np.sum(state_codes == 7),
            '000 (不需要校正)': np.sum(state_codes == 0),
        }

        # 4. 漂亮的打印輸出
        print(f"\n{'='*10} Correction Analysis {'='*10}")
        print(f"Total Samples: {len(y_clean)}")
        
        print(f"\n[🟢 Success - 雜訊成功被移除]")
        print(f"  101 (原本是1, 變成0, 修回1): {stats['101 (校正對了)']}")
        print(f"  010 (原本是0, 變成1, 修回0): {stats['010 (校正對了)']}")
        
        print(f"\n[🔴 Damage - 乾淨資料被改壞]")
        print(f"  110 (原本是1, 沒變, 卻改成0): {stats['110 (校正錯了)']}")
        print(f"  001 (原本是0, 沒變, 卻改成1): {stats['001 (校正錯了)']}")
        
        print(f"\n[⚠️ Missed - 雜訊依然存在]")
        print(f"  100 (原本是1, 變成0, 沒修回來): {stats['100 (沒校正到)']}")
        print(f"  011 (原本是0, 變成1, 沒修回來): {stats['011 (沒校正到)']}")
        
        print(f"\n[⚪ Keep - 正確保持不動]")
        print(f"  111 & 000 (資料原本乾淨且未被修改): {stats['111 (不需要校正)'] + stats['000 (不需要校正)']}")
        print(f"{'='*40}\n")

        return y_corrected_soft
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
            logger.info(f"  [Label {label} Correction Report]")
            logger.info(f"    - Original Positives: {num_pos}, Negatives: {num_neg}")
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
        print(f"[{self.__class__.__name__}] Running Per-Label GMM Correction...")
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
            
            # 只有當該類別的正樣本夠多時，才跑 GMM (避免 sample 太少 GMM 炸開)
            target_watch_list = [0,1,3,4,5,6,8,9,11,12,13,14]  # 你可以指定一些類別來畫圖檢查
            if len(pos_scores_c) > 1:
                prob_clean = self._run_gmm_on_subset(pos_scores_c)
                if c in target_watch_list:
                    # 顯示結果
                    visualize_gmm(pos_scores_c, class_name=c, subset_type=f"Pos_Original{args.alpha}",save_dir=f"gmm_debug_plots{args.alpha}")
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
                stats['pos_origin'] = len(pos_scores_c)
                stats["pos_flipped_to_neg"] += flipped_0
                stats["soft_labels"] += soft
            # -------------------------------------------
            # (B) 該類別的負樣本 (Negative) 校正
            neg_scores_c = col_scores[neg_mask]
            if len(neg_scores_c) > 1:
                prob_clean = self._run_gmm_on_subset(neg_scores_c)
                if c in target_watch_list:

                    visualize_gmm(neg_scores_c, class_name=c, subset_type=f"Neg_Original{args.alpha}",save_dir=f"gmm_debug_plots{args.alpha}")
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
                stats['neg_origin'] =len(neg_scores_c)
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
    
# 負責提取分數並篩選雜訊的 Pipeline   
class NoiseSelectionPipeline:
    def __init__(self, calculator: BaseScoreCalculator, filter_strategy: BaseNoiseFilter):
        self.calculator = calculator
        self.filter_strategy = filter_strategy

    def run(self, model, dataloader, device):
        print("1. Calculating Scores...")
        indices, scores,labels = self.calculator.calculate(model, dataloader, device)
        
        print("2. Filtering Noise...")
        clean_ids, noisy_ids = self.filter_strategy.filter(scores, indices)
        
        return clean_ids, noisy_ids, scores, labels

class HSMHybridPipeline:
    """
    專門處理 HSM 這種需要結合兩種分數的複雜流程
    這是一個更高階的 Orchestrator
    """
    def __init__(self, rel_calculator, cd_calculator, filter_strategy, alpha=0.5):
        self.rel_calc = rel_calculator
        self.cd_calc = cd_calculator
        self.filter_strategy = filter_strategy
        self.alpha = alpha

    def run(self, model, dataloader, device):
        print("🚀 Starting HSM Hybrid Selection...")
        
        # 1. 分別計算兩種分數
        idx1, rel_scores,labels = self.rel_calc.calculate(model, dataloader, device)
        # 注意: 這裡假設 dataloader 順序固定，idx1 應該等於 idx2。
        # 嚴謹作法應該要做 index matching，這裡簡化處理。
        _, cd_scores,_= self.cd_calc.calculate(model, dataloader, device)
        
        # 2. 診斷 (Optional)
        print(f"  - REL Max: {rel_scores.max():.4f}, Mean: {rel_scores.mean():.4f}")
        print(f"  - CD Max: {cd_scores.max():.4f}, Mean: {cd_scores.mean():.4f}")
        
        # 3. 歸一化
        rel_norm = MathUtils.minmax_normalize_columnwise(rel_scores)
        cd_norm = MathUtils.minmax_normalize_columnwise(cd_scores)
        
        # 4. 融合 (Fusion)
        final_score = rel_norm * self.alpha + cd_norm * (1 - self.alpha)
        final_scores = final_score
        clean_ids, noisy_ids = self.filter_strategy.filter(final_scores, idx1)
        # 5. 篩選
        return clean_ids, noisy_ids, final_scores, labels


    def run_score_only(self, model, dataloader, device):
        """只計算並融合分數，不進行篩選/校正 (為了外部參數搜尋用)"""
        print("🚀 Pipeline: Calculating Scores Only...")
        
        # 1. 計算
        idx1, rel_scores, labels = self.rel_calc.calculate(model, dataloader, device)
        _, cd_scores, _ = self.cd_calc.calculate(model, dataloader, device)
        
        # 2. 歸一化
        rel_norm = MathUtils.minmax_normalize_columnwise(rel_scores)
        cd_norm = MathUtils.minmax_normalize_columnwise(cd_scores)
        
        # 3. 融合
        hsm_scores = rel_norm * self.alpha + cd_norm * (1 - self.alpha)
        
        # 回傳分數，讓外部迴圈去決定怎麼切
        return hsm_scores, labels, idx1
# 