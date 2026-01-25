import torch
import pandas as pd
import os
from typing import List, Dict, Any
# from util.seen import get_noise_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import os
import numpy as np
from scipy.stats import norm
class NoiseCorrectionEvaluator:
    """
    負責計算特定 Label 的各項統計數據 (TP, TN, Confusion Matrix)。
    符合 SRP：只負責計算，不負責存檔或流程控制。
    """
    def __init__(self, label_names: List[str]):
        self.label_names = label_names

    def compute_label_stats(self, args,
                        label_index: int, 
                        y_true: torch.Tensor, 
                        y_noisy: torch.Tensor, 
                        y_corrected: torch.Tensor, 
                        method_name: str) -> Dict[str, Any]:
    
        label_name = self.label_names[label_index]
        
        # 1. 統一轉 Float
        if isinstance(y_corrected, torch.Tensor):
            C_float = y_corrected[:, label_index].float().cpu()
        else:
            C_float = torch.tensor(y_corrected[:, label_index]).float().cpu()

        T = y_true[:, label_index].long().cpu()
        N = y_noisy[:, label_index].long().cpu()
        
        # ==========================================
        #   A. 區分 Hard 與 Soft
        # ==========================================
        eps = args.epsilon
        is_soft = (C_float > eps) & (C_float < (1.0 - eps))
        is_hard = ~is_soft

        # ==========================================
        #   B. 計算 8-State Code (基於全體)
        # ==========================================
        C_binary = C_float.round().long()
        state_codes = (T << 2) | (N << 1) | C_binary
        
        state_names = {
            5: 'Fix_FN',  2: 'Fix_FP',
            6: 'Broke_TP', 1: 'Broke_TN',
            4: 'Miss_FN', 3: 'Miss_FP',
            7: 'Keep_TP', 0: 'Keep_TN'
        }

        # ==========================================
        #   C. 詳細統計 (Hard vs Soft)
        # ==========================================
        stats_breakdown = {}
        
        # 用來檢查總數的累加器
        total_samples_check = 0
        
        for code in range(8):
            name = state_names[code]
            in_this_state = (state_codes == code)
            
            count_hard = (in_this_state & is_hard).sum().item()
            count_soft = (in_this_state & is_soft).sum().item()
            
            total_samples_check += (count_hard + count_soft)

            stats_breakdown[f'{name}_Hard'] = count_hard
            stats_breakdown[f'{name}_Soft'] = count_soft
            
            # Soft Confidence (Optional)
            if count_soft > 0:
                stats_breakdown[f'{name}_Soft_Conf'] = C_float[in_this_state & is_soft].mean().item()
            else:
                stats_breakdown[f'{name}_Soft_Conf'] = 0.0

        # ==========================================
        #   D. 計算指標 (僅針對 Hard 樣本)
        # ==========================================
        def calc_scores(y_t, y_p):
            if len(y_t) == 0: return 0.0, 0.0
            tp = ((y_t == 1) & (y_p == 1)).sum().item()
            fp = ((y_t == 0) & (y_p == 1)).sum().item()
            fn = ((y_t == 1) & (y_p == 0)).sum().item()
            tn = ((y_t == 0) & (y_p == 0)).sum().item() # 順便算 TN
            
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            return rec, prec, tp, fp, tn, fn

        rec_hard, prec_hard, tp_h, fp_h, tn_h, fn_h = calc_scores(T[is_hard], C_binary[is_hard])

        # ==========================================
        #   E. 總數驗證與矩陣轉換 (關鍵修正!)
        # ==========================================
        # 這裡幫你把 8-State 自動歸類回 TP/TN/FP/FN，確保總數正確
        
        # 1. 計算 "最終" 的 Confusion Matrix (包含 Hard + Soft)
        # 這是根據 C_binary (四捨五入後) 的結果
        final_TN = ((state_codes == 0) | (state_codes == 2)).sum().item() # Keep_TN + Fix_FP
        final_FP = ((state_codes == 1) | (state_codes == 3)).sum().item() # Broke_TN + Miss_FP
        final_TP = ((state_codes == 7) | (state_codes == 5)).sum().item() # Keep_TP + Fix_FN
        final_FN = ((state_codes == 6) | (state_codes == 4)).sum().item() # Broke_TP + Miss_FN
        
        # 2. 驗證是否吻合 Ground Truth
        true_pos_count = (T == 1).sum().item()
        true_neg_count = (T == 0).sum().item()
        
        # 應該要相等： (TN + FP) 必須等於 真實負樣本數
        assert (final_TN + final_FP) == true_neg_count, \
            f"TN({final_TN}) + FP({final_FP}) != True_Neg_Count({true_neg_count}). Check logic!"

        return {
            'label_index': label_index,
            'label_name': label_name,
            'method': method_name,
            
            # 1. 樣本分佈概況
            'total_count': len(T),
            'hard_count': is_hard.sum().item(),
            'soft_count': is_soft.sum().item(),
            
            # 2. 詳細 8-State 數據
            **stats_breakdown,
            
            # 3. Hard 樣本指標
            'precision_hard': prec_hard,
            'recall_hard': rec_hard,
            
            # 4. [新增] 最終預測結果統計 (這是你要檢查總數的地方)
            'Final_TN_Count': final_TN, # 包含 Keep_TN, Fix_FP
            'Final_FP_Count': final_FP, # 包含 Broke_TN, Miss_FP
            'Final_TP_Count': final_TP,
            'Final_FN_Count': final_FN,
            
            # 5. 真實分佈 (作為對照)
            'True_Neg_Count': true_neg_count,
            'True_Pos_Count': true_pos_count
        }
class ResultRecorder:
    """
    負責收集統計結果並寫入檔案。
    符合 OCP：若未來要改存 JSON 或 Database，只需擴充或修改此類別，不影響主邏輯。
    """
    def __init__(self, result_dir: str):
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.records = []

    def add_record(self, record: Dict[str, Any]):
        self.records.append(record)

    def clear_records(self):
        self.records = []

    def save_to_csv(self, filename: str):
        df_stats = pd.DataFrame(self.records)
        save_path = os.path.join(self.result_dir, filename)
        df_stats.to_csv(save_path, index=False)
        return save_path
    
def debug_plot_gmm(data, class_name, subset_type, save_dir="debug_plots"):
    """
    data: 分數陣列 (e.g., pos_scores_c)
    class_name: 類別名稱或索引 (e.g., 'stat.ML' 或 5)
    subset_type: 'Positive_Subset' or 'Negative_Subset'
    """
    import json 
    with open('dataset/AAPD/label_to_index.json', 'r') as f:
        label_to_index = json.load(f)
    index_to_label = {v:k for k,v in label_to_index.items()}
    if len(data) < 5: return # 樣本太少不畫
    
    # 確保目錄存在
    os.makedirs(save_dir, exist_ok=True)

    # 為了畫圖，重新 fit 一個簡單的 GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    data_reshape = data.reshape(-1, 1)
    gmm.fit(data_reshape)
    
    # 取得參數
    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    
    plt.figure(figsize=(8, 4))
    
    # 1. 畫原始數據直方圖
    sns.histplot(data, bins=30, stat='density', alpha=0.5, color='gray', label='Score Dist')
    
    # 2. 標示 GMM 的兩個中心
    plt.axvline(x=means[0], color='r', linestyle='--', label=f'Mean 1: {means[0]:.2f} (w={weights[0]:.2f})')
    plt.axvline(x=means[1], color='b', linestyle='--', label=f'Mean 2: {means[1]:.2f} (w={weights[1]:.2f})')
    # class name mapping
    if isinstance(class_name, int):
        class_name = index_to_label.get(class_name, str(class_name))
    plt.title(f"GMM Check: Class {class_name} ({subset_type})")
    plt.legend()
    
    # 存檔而不是 show (避免迴圈卡住)
    filename = f"{save_dir}/gmm_{class_name}_{subset_type}.png"
    plt.savefig(filename)
    plt.close() # 關閉畫布釋放記憶體
    print(f"  [Plot Saved] {filename}")
def visualize_gmm(data, class_name, subset_type, save_dir="gmm_debug_plots"):
    """
    功能：繪製 GMM 雙峰圖 (直方圖 + 綠色Clean曲線 + 紅色Noise曲線)
    """
    # 1. 基本檢查與目錄建立
    if len(data) < 10: return # 樣本太少不畫
    os.makedirs(save_dir, exist_ok=True)
    
    data = np.asarray(data).reshape(-1, 1)

    # 2. 為了畫圖，我們在這邊快速重新 Fit 一次 GMM
    # (這樣最安全，確保畫出來的線跟數據是匹配的)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data)

    # 3. 取得參數並排序 (Mean 小的是 Clean，大的是 Noise)
    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    covariances = gmm.covariances_.flatten()
    stds = np.sqrt(covariances)

    idx = np.argsort(means) # 排序索引
    means, weights, stds = means[idx], weights[idx], stds[idx]

    # 4. 開始繪圖
    plt.figure(figsize=(8, 5))
    
    # (A) 畫原始數據直方圖 (Histogram)
    sns.histplot(data, bins=40, stat='density', alpha=0.3, color='gray', label='Score Distribution')

    # (B) 準備 X 軸
    x_min, x_max = data.min(), data.max()
    x_axis = np.linspace(x_min, x_max, 1000)

    # (C) 畫 Clean Component (綠色)
    y_clean = weights[0] * norm.pdf(x_axis, means[0], stds[0])
    plt.plot(x_axis, y_clean, color='green', linewidth=2, linestyle='--', label=f'Clean (Mean={means[0]:.2f})')
    plt.fill_between(x_axis, y_clean, color='green', alpha=0.1) # 填色

    # (D) 畫 Noise Component (紅色)
    y_noise = weights[1] * norm.pdf(x_axis, means[1], stds[1])
    plt.plot(x_axis, y_noise, color='red', linewidth=2, linestyle='--', label=f'Noise (Mean={means[1]:.2f})')
    plt.fill_between(x_axis, y_noise, color='red', alpha=0.1)
    
    # label id to index
    import json 
    with open('dataset/AAPD/label_to_index.json', 'r') as f:
        label_to_index = json.load(f)
    index_to_label = {v:k for k,v in label_to_index.items()}
    if isinstance(class_name, int):
        class_name = index_to_label.get(class_name, str(class_name))
    # (E) 裝飾圖片
    plt.title(f"Class:{class_name} ({subset_type}) ")
    plt.xlabel("HSM Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 存檔
    filename = f"{save_dir}/class_{class_name}_{subset_type}.png"
    plt.savefig(filename)
    plt.close() # 關閉畫布，釋放記憶體
    print(f"  [Debug] GMM Plot saved to {filename}")



def get_noise_confusion_matrix(y_true: np.ndarray,
                               y_noisy: np.ndarray) -> pd.DataFrame:
    """
    生成一個 2x2 的混淆矩陣，比較真實標籤 (T) 和噪聲標籤 (N)。

    Args:
        y_true (np.ndarray): 真實標籤 (N, L)
        y_noisy (np.ndarray): 帶噪聲的標籤 (N, L)

    Returns:
        pd.DataFrame: 包含 2x2 狀態計數的 Pandas DataFrame。
    """

    # 1. 計算四個象限的計數
    # (T=0, N=0)
   
    tn_count = torch.sum((y_true == 0) & (y_noisy == 0)).item()
    # True=0, Pred=1 (FP) -> 多標記 (Extra Noise)
    fp_count = torch.sum((y_true == 0) & (y_noisy == 1)).item()
    # True=1, Pred=0 (FN) -> 少標記 (Missing Label)
    fn_count = torch.sum((y_true == 1) & (y_noisy == 0)).item()
    # True=1, Pred=1 (TP)
    tp_count = torch.sum((y_true == 1) & (y_noisy == 1)).item()

    # 2. 建立 2x2 矩陣
    matrix = np.array([
        [tp_count, fn_count],
        [fp_count, tn_count]
    ])

    # 3. 使用 Pandas 進行格式化輸出
    df = pd.DataFrame(
        matrix,
        columns=["Noisy (N) = 1", "Noisy (N) = 0"],
        index=[
            "True (T) = 1",
            "True (T) = 0"
        ]
    )