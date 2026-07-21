import torch
import pandas as pd
import os
from typing import List, Dict, Any
# from util.seen import get_noise_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
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
        eps = args.epsilon
        # 1. 統一轉 Float
        if isinstance(y_corrected, torch.Tensor):
            C_float = y_corrected[:, label_index].float().cpu()
        else:
            C_float = torch.tensor(y_corrected[:, label_index]).float().cpu()

        T = y_true[:, label_index].long().cpu()
        N = y_noisy[:, label_index].long().cpu()
        
        C_binary = C_float.round().long()
        state_codes = (T << 2) | (N << 1) | C_binary
        # ==========================================
        #   A. 區分 Hard 與 Soft
        # ==========================================
        
        is_soft = (C_float > eps) & (C_float < (1.0 - eps))
        is_hard = ~is_soft

        # ==========================================
        #   B. 計算 8-State Code (基於全體)
        # ==========================================
       
        
        state_names = [
        'Keep_TN',  # 0: 000
        'Broke_TN', # 1: 001
        'Fix_FP',   # 2: 010
        'Miss_FP',  # 3: 011
        'Miss_FN',  # 4: 100
        'Fix_FN',   # 5: 101
        'Broke_TP', # 6: 110
        'Keep_TP'   # 7: 111
    ]
        # ==============================ㄤㄤ============
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
            if len(y_t) == 0: return 0.0, 0.0, 0, 0, 0, 0
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

        # 被篩選掉的樣本 = 從 N=1 翻轉為 0 的樣本 (Fix_FP + Broke_TP)
        filtered_count = stats_breakdown['Fix_FP_Hard'] + stats_breakdown['Broke_TP_Hard']
        noisy_pos_count = (N == 1).sum().item()
        filtered_ratio = filtered_count / noisy_pos_count if noisy_pos_count > 0 else 0.0

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

            # 4. 最終預測結果統計
            'Final_TN_Count': final_TN,
            'Final_FP_Count': final_FP,
            'Final_TP_Count': final_TP,
            'Final_FN_Count': final_FN,

            # 5. 真實分佈
            'True_Neg_Count': true_neg_count,
            'True_Pos_Count': true_pos_count,

            # 6. 篩選比例：被翻轉(1→0)的樣本佔所有 N=1 樣本的比例
            'filtered_ratio': filtered_ratio,
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
    
def debug_plot_gmm(data, class_name, subset_type, save_dir="debug_plots", label_index_path='dataset/AAPD/label_to_index.json'):
    """
    data: 分數陣列 (e.g., pos_scores_c)
    class_name: 類別名稱或索引 (e.g., 'stat.ML' 或 5)
    subset_type: 'Positive_Subset' or 'Negative_Subset'
    """
    import json
    with open(label_index_path, 'r') as f:
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
def visualize_gmm(data, class_name, subset_type, save_dir="gmm_debug_plots", n_components=2, label_index_path='dataset/AAPD/label_to_index.json', encoder_name="", use_count=True):
    """
    功能：繪製 GMM 分佈圖 (直方圖 + 高斯分佈曲線)

    Args:
        data: 分數陣列
        class_name: 類別名稱或索引
        subset_type: 子集類型 (e.g., 'Pos_Original', 'Neg_Original')
        save_dir: 儲存目錄
        n_components: GMM 的 component 數量 (2 或 3)
        use_count: True 使用真實計數，False 使用密度 (預設 False)
    """
    # 1. 基本檢查與目錄建立
    if len(data) < 10: return # 樣本太少不畫
    os.makedirs(save_dir, exist_ok=True)

    data_flat = np.asarray(data).flatten()
    data_reshaped = data_flat.reshape(-1, 1)

    # 2. 為了畫圖，我們在這邊快速重新 Fit 一次 GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data_reshaped)

    # 3. 取得參數並排序 (Mean 小的是 Clean，大的是 Noise)
    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    covariances = gmm.covariances_.flatten()
    stds = np.sqrt(covariances)

    idx = np.argsort(means)
    means, weights, stds = means[idx], weights[idx], stds[idx]

    # 4. 開始繪圖
    plt.figure(figsize=(10, 6))

    # (A) 畫原始數據直方圖 (Histogram)
    stat_mode = 'count' if use_count else 'density'
    sns.histplot(data_reshaped, bins=40, stat=stat_mode, alpha=0.3, color='gray', label='Score Distribution')

    # (B) 準備 X 軸
    x_min, x_max = data_flat.min(), data_flat.max()
    x_axis = np.linspace(x_min, x_max, 1000)

    # (C) 計算高斯曲線的縮放因子
    if use_count:
        # 計數模式：將密度函數乘以 bin width 和總樣本數
        n_bins = 40
        bin_width = (x_max - x_min) / n_bins
        scale_factor = len(data_flat) * bin_width
    else:
        # 密度模式：保持原本的權重（和為 1）
        scale_factor = 1.0

    # (D) 根據 n_components 繪製不同數量的曲線
    def plot_component(color, label_name, mean, weight, std, idx_order):
        if use_count:
            y = weight * scale_factor * norm.pdf(x_axis, mean, std)
        else:
            y = weight * norm.pdf(x_axis, mean, std)
        plt.plot(x_axis, y, color=color, linewidth=2.5, linestyle='--',
                 label=f'{label_name} (μ={mean:.3f}, w={weight:.3f})')
        plt.fill_between(x_axis, y, color=color, alpha=0.15)

    if n_components == 2:
        plot_component('green', 'Clean', means[0], weights[0], stds[0], 0)
        plot_component('red', 'Noise', means[1], weights[1], stds[1], 1)

    elif n_components == 3:
        plot_component('green', 'Clean', means[0], weights[0], stds[0], 0)
        plot_component('orange', 'Mid', means[1], weights[1], stds[1], 1)
        plot_component('red', 'Noise', means[2], weights[2], stds[2], 2)

    elif n_components == 4:
        comp_configs = [
            ('green', 'Clean', 0),
            ('yellowgreen', 'Mid-Clean', 1),
            ('orange', 'Mid-Noisy', 2),
            ('red', 'Noisy', 3),
        ]
        for color, label_name, i in comp_configs:
            plot_component(color, label_name, means[i], weights[i], stds[i], i)

    elif n_components == 5:
        comp_configs = [
            ('green', 'Clean', 0),
            ('yellowgreen', 'Mid-Clean', 1),
            ('gold', 'Uncertain', 2),
            ('orange', 'Mid-Noisy', 3),
            ('red', 'Noisy', 4),
        ]
        for color, label_name, i in comp_configs:
            plot_component(color, label_name, means[i], weights[i], stds[i], i)

    # (E) 標籤轉換
    import json
    try:
        with open(label_index_path, 'r') as f:
            label_to_index = json.load(f)
        index_to_label = {v:k for k,v in label_to_index.items()}
        if isinstance(class_name, int):
            class_name = index_to_label.get(class_name, str(class_name))
    except:
        pass

    # (F) 裝飾圖片
    mode_str = "Count" if use_count else "Density"
    plt.title(f"Class:{class_name} ({subset_type}) - {n_components} Components [{mode_str}]")
    plt.xlabel("HSM Score")
    plt.ylabel(f"{'Count' if use_count else 'Density'}")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 5. 存檔
    suffix = "_count" if use_count else "_density"
    filename = f"{save_dir}/class_{class_name}_{encoder_name}_{subset_type}_{n_components}{suffix}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Debug] GMM Plot ({mode_str}) saved to {filename}")


def visualize_gmm_comparison(data, class_name, subset_type, save_dir="gmm_debug_plots", n_components=2, label_index_path='dataset/AAPD/label_to_index.json', encoder_name=""):
    """
    並排繪製 Count 和 Density 兩種模式的 GMM 分佈圖，方便對比差異。

    Args:
        data: 分數陣列
        class_name: 類別名稱或索引
        subset_type: 子集類型
        save_dir: 儲存目錄
        n_components: GMM 的 component 數量
        label_index_path: 標籤索引文件路徑
        encoder_name: 編碼器名稱
    """
    if len(data) < 10:
        return

    os.makedirs(save_dir, exist_ok=True)

    data_flat = np.asarray(data).flatten()
    data_reshaped = data_flat.reshape(-1, 1)

    # 重新 Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data_reshaped)

    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    covariances = gmm.covariances_.flatten()
    stds = np.sqrt(covariances)

    idx = np.argsort(means)
    means, weights, stds = means[idx], weights[idx], stds[idx]

    # 準備數據
    x_min, x_max = data_flat.min(), data_flat.max()
    x_axis = np.linspace(x_min, x_max, 1000)
    n_bins = 40
    bin_width = (x_max - x_min) / n_bins

    # 並排圖表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 顏色配置
    comp_colors = {
        2: ['green', 'red'],
        3: ['green', 'orange', 'red'],
        4: ['green', 'yellowgreen', 'orange', 'red'],
        5: ['green', 'yellowgreen', 'gold', 'orange', 'red'],
    }
    colors = comp_colors.get(n_components, ['blue'] * n_components)

    # ========== LEFT: Density 模式 ==========
    ax = axes[0]
    sns.histplot(data_reshaped, bins=n_bins, stat='density', alpha=0.3, color='gray', ax=ax, label='Distribution')

    for i in range(n_components):
        y = weights[i] * norm.pdf(x_axis, means[i], stds[i])
        ax.plot(x_axis, y, color=colors[i], linewidth=2.5, linestyle='--',
                label=f'Comp{i+1} (μ={means[i]:.3f})')
        ax.fill_between(x_axis, y, color=colors[i], alpha=0.15)

    ax.set_title(f"Density Mode - {n_components} Components", fontsize=12, fontweight='bold')
    ax.set_xlabel("HSM Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== RIGHT: Count 模式 ==========
    ax = axes[1]
    sns.histplot(data_reshaped, bins=n_bins, stat='count', alpha=0.3, color='gray', ax=ax, label='Distribution')

    scale_factor = len(data_flat) * bin_width

    for i in range(n_components):
        y = weights[i] * scale_factor * norm.pdf(x_axis, means[i], stds[i])
        ax.plot(x_axis, y, color=colors[i], linewidth=2.5, linestyle='--',
                label=f'Comp{i+1} (μ={means[i]:.3f})')
        ax.fill_between(x_axis, y, color=colors[i], alpha=0.15)

    ax.set_title(f"Count Mode - {n_components} Components", fontsize=12, fontweight='bold')
    ax.set_xlabel("HSM Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 整體標題
    import json
    try:
        with open(label_index_path, 'r') as f:
            label_to_index = json.load(f)
        index_to_label = {v:k for k,v in label_to_index.items()}
        if isinstance(class_name, int):
            class_name = index_to_label.get(class_name, str(class_name))
    except:
        pass

    fig.suptitle(f"Class: {class_name} ({subset_type}) - Density vs Count Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 存檔
    filename = f"{save_dir}/comparison_class_{class_name}_{encoder_name}_{subset_type}_{n_components}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Comparison] GMM Plot saved to {filename}")


def visualize_gmm_cluster_purity_k(
    cd_vals: np.ndarray,
    gap_vals: np.ndarray,
    true_labels: np.ndarray,
    label_index: int,
    k_values=(2, 3, 4, 5),
    save_dir: str = "gmm_debug_plots",
    label_index_path: str = "dataset/AAPD/label_to_index.json",
    encoder_name: str = "",
):
    """
    對 label=1 的候選池，分別用 k=2,3,4,5 個 GMM 群，
    分析「最高可疑群」(mean norm 最大) 中 TP/FP 佔比的逐步變化。

    Args:
        cd_vals:     [n] CD 分數（label=1 pool，masked，範圍 [-1, 0]）
        gap_vals:    [n] Positive Gap 分數（label=1 pool，正值）
        true_labels: [n] 真實標籤，1=TP, 0=FP
        label_index: 標籤索引（用於命名）
        k_values:    要嘗試的 k 值列表
        save_dir:    儲存目錄
        label_index_path: 標籤名稱映射路徑
    """
    import json
    from sklearn.mixture import GaussianMixture

    if len(cd_vals) < max(k_values) + 1:
        print(f"  [PurityK] Label {label_index}: samples too few ({len(cd_vals)}), skip.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 標籤名稱解析
    label_name = str(label_index)
    try:
        with open(label_index_path, 'r') as f:
            label_to_index = json.load(f)
        index_to_label = {v: k for k, v in label_to_index.items()}
        label_name = index_to_label.get(label_index, str(label_index))
    except Exception:
        pass

    # 2D feature: (-CD, Gap)，方向一致，值越大代表 FP 嫌疑越高
    x1 = -cd_vals
    x2 = gap_vals
    X = np.column_stack([x1, x2])

    n_k = len(k_values)
    # 上方: 每個 k 的「各群」TP/FP 堆疊 bar chart
    # 下方: 每個 k 的散點圖，顏色=cluster, 形狀=TP/FP
    fig, axes = plt.subplots(2, n_k, figsize=(5 * n_k, 10))
    if n_k == 1:
        axes = axes.reshape(2, 1)

    # summary_rows: list of (k, per_cluster_stats)
    # per_cluster_stats: list of dict {cid, tp, fp, total, fp_pct, is_noisy}
    summary_rows = []

    for col_idx, k in enumerate(k_values):
        try:
            gmm = GaussianMixture(n_components=k, max_iter=200, tol=1e-3,
                                  reg_covar=1e-4, random_state=0)
            gmm.fit(X)
        except Exception as e:
            print(f"  [PurityK] k={k} GMM failed: {e}")
            continue

        # 根據 mean norm 升序排列：0=最乾淨, k-1=最可疑
        mean_norms = np.linalg.norm(gmm.means_, axis=1)
        sorted_idx = np.argsort(mean_norms)
        cluster_assignments = gmm.predict(X)
        remap = {old: new for new, old in enumerate(sorted_idx)}
        assignments = np.array([remap[a] for a in cluster_assignments])
        noisy_cluster_id = k - 1

        # 計算每群的 TP/FP 數量
        per_cluster = []
        for cid in range(k):
            mask = (assignments == cid)
            tp = int(np.sum(true_labels[mask] == 1))
            fp = int(np.sum(true_labels[mask] == 0))
            total = tp + fp
            fp_pct = fp / total if total > 0 else 0.0
            per_cluster.append({
                'cid': cid, 'tp': tp, 'fp': fp,
                'total': total, 'fp_pct': fp_pct,
                'is_noisy': (cid == noisy_cluster_id),
            })
        summary_rows.append((k, per_cluster))

        # ── 上方: 各群 TP/FP 堆疊 bar chart ──────────────────────
        ax_bar = axes[0, col_idx]
        x_pos = np.arange(k)
        tp_counts = [d['tp'] for d in per_cluster]
        fp_counts = [d['fp'] for d in per_cluster]
        totals    = [d['total'] for d in per_cluster]

        ax_bar.bar(x_pos, tp_counts, label='TP', color='steelblue',
                   alpha=0.85, edgecolor='black')
        ax_bar.bar(x_pos, fp_counts, bottom=tp_counts, label='FP',
                   color='tomato', alpha=0.85, edgecolor='black')

        # 在每個 bar 上標注 FP% 與總數，最可疑群加框
        for cid, (tp, fp, tot) in enumerate(zip(tp_counts, fp_counts, totals)):
            fp_pct = fp / tot * 100 if tot > 0 else 0
            ax_bar.text(cid, tot + 0.3,
                        f"FP:{fp_pct:.0f}%\n(n={tot})",
                        ha='center', va='bottom', fontsize=8,
                        color='darkred' if cid == noisy_cluster_id else 'black',
                        fontweight='bold' if cid == noisy_cluster_id else 'normal')

        # 最高可疑群的 x-tick 加星號
        xtick_labels = [
            f"C{cid}\n★most" if cid == noisy_cluster_id else f"C{cid}"
            for cid in range(k)
        ]
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(xtick_labels, fontsize=8)
        ax_bar.set_ylabel("Count")
        ax_bar.set_title(f"k={k}  各群 TP/FP 組成", fontsize=11)
        ax_bar.legend(fontsize=8, loc='upper right')
        ax_bar.grid(True, alpha=0.3, axis='y')

        # ── 下方: 散點圖 ──────────────────────────────────────────
        ax_sc = axes[1, col_idx]
        cmap = plt.cm.get_cmap('tab10', k)
        for cid in range(k):
            mask = (assignments == cid)
            tp_m = mask & (true_labels == 1)
            fp_m = mask & (true_labels == 0)
            color = cmap(cid)
            star = "★" if cid == noisy_cluster_id else ""
            label_str = f"C{cid}{star} TP={tp_counts[cid]} FP={fp_counts[cid]}"
            ax_sc.scatter(x1[tp_m], x2[tp_m], marker='o', s=18,
                          color=color, alpha=0.6)
            ax_sc.scatter(x1[fp_m], x2[fp_m], marker='x', s=30,
                          color=color, alpha=0.9)
            # legend proxy
            ax_sc.scatter([], [], marker='s', color=color, label=label_str, s=40)

        ax_sc.set_xlabel("-CD (FP嫌疑↑)")
        ax_sc.set_ylabel("Positive Gap (FP嫌疑↑)")
        ax_sc.set_title(f"k={k} 群分布  (○=TP  ×=FP)", fontsize=10)
        ax_sc.legend(fontsize=7, loc='upper left')
        ax_sc.grid(True, alpha=0.3)

    # ── Console 摘要 + CSV 存檔 ──────────────────────────────────
    print(f"\n  [PurityK] Label '{label_name}' ({encoder_name})  總樣本={len(cd_vals)}")
    header = f"  {'k':>3}  {'Cluster':>8}  {'TP':>6}  {'FP':>6}  {'Total':>6}  {'FP%':>7}  Note"
    print(header)
    print("  " + "-" * (len(header) - 2))

    csv_rows = []
    for k, per_cluster in summary_rows:
        for d in per_cluster:
            note = "most_suspicious" if d['is_noisy'] else ""
            print(f"  {k:>3}  {'C'+str(d['cid']):>8}  "
                  f"{d['tp']:>6}  {d['fp']:>6}  {d['total']:>6}  "
                  f"{d['fp_pct']*100:>6.1f}%"
                  + (f"  ← {note}" if note else ""))
            csv_rows.append({
                'label_index':   label_index,
                'label_name':    label_name,
                'encoder':       encoder_name,
                'total_pool':    len(cd_vals),
                'k':             k,
                'cluster_id':    d['cid'],
                'cluster_rank':  f"C{d['cid']}({'most' if d['is_noisy'] else 'clean' if d['cid']==0 else 'mid'})",
                'is_most_suspicious': int(d['is_noisy']),
                'TP':            d['tp'],
                'FP':            d['fp'],
                'Total':         d['total'],
                'FP_pct':        round(d['fp_pct'] * 100, 2),
                'TP_pct':        round((1 - d['fp_pct']) * 100, 2),
            })
        print()

    # 存 CSV
    csv_df = pd.DataFrame(csv_rows)
    csv_fname = f"{save_dir}/purity_k_{label_name}_{encoder_name}.csv"
    csv_df.to_csv(csv_fname, index=False)
    print(f"  [PurityK] Table saved: {csv_fname}")

    # 存圖
    fig.suptitle(
        f"Label: {label_name} [{encoder_name}]  各群 TP/FP 逐群分析\n"
        f"(★=最高可疑群, ○=TP, ×=FP, 總樣本={len(cd_vals)})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    img_fname = f"{save_dir}/purity_k_{label_name}_{encoder_name}.png"
    plt.savefig(img_fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [PurityK] Plot  saved: {img_fname}")


def visualize_1d_gmm_features(
    cd_vals: np.ndarray,
    gap_vals: np.ndarray,
    true_labels: np.ndarray,
    label_index: int,
    n_components: int = 2,
    save_dir: str = "gmm_debug_plots",
    label_index_path: str = "dataset/AAPD/label_to_index.json",
    encoder_name: str = "",
):
    """
    對 label=1 候選池，分別對 -CD 和 Positive Gap 做 1D GMM，
    並顯示每個 cluster 中 TP/FP 的分布（堆疊直方圖 + GMM 曲線）。

    Args:
        cd_vals:     [n] CD 分數（label=1，範圍 [-1, 0]，越靠近 0 = FP 嫌疑越高）
        gap_vals:    [n] Positive Gap 分數（label=1，正值，越大 = FP 嫌疑越高）
        true_labels: [n] 真實標籤，1=TP, 0=FP
        label_index: 標籤索引
        n_components: GMM 群數 (2..5)
        save_dir:    儲存目錄
        label_index_path: 標籤名稱映射路徑
    """
    import json
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm as scipy_norm

    if len(cd_vals) < n_components + 1:
        print(f"  [1DGMM] Label {label_index}: samples too few ({len(cd_vals)}), skip.")
        return

    os.makedirs(save_dir, exist_ok=True)

    label_name = str(label_index)
    try:
        with open(label_index_path, 'r') as f:
            label_to_index = json.load(f)
        index_to_label = {v: k for k, v in label_to_index.items()}
        label_name = index_to_label.get(label_index, str(label_index))
    except Exception:
        pass

    # 配置顏色
    comp_colors = {
        2: ['green', 'red'],
        3: ['green', 'orange', 'red'],
        4: ['green', 'yellowgreen', 'orange', 'red'],
        5: ['green', 'yellowgreen', 'gold', 'orange', 'red'],
    }
    colors = comp_colors.get(n_components, ['blue'] * n_components)

    def _fit_and_plot_1d(ax_hist, ax_bar, scores_1d, feature_label):
        """
        在給定的軸上：
        - ax_hist: 1D GMM histogram + curve，區分 TP (實線) / FP (虛線)
        - ax_bar:  每個 cluster 的 TP/FP 堆疊 bar chart
        """
        data = scores_1d.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, max_iter=200, tol=1e-3,
                              reg_covar=1e-4, random_state=0)
        gmm.fit(data)

        # 按 mean 升序排列（小 mean = clean）
        idx_sorted = np.argsort(gmm.means_.flatten())
        means = gmm.means_.flatten()[idx_sorted]
        stds = np.sqrt(gmm.covariances_.flatten()[idx_sorted])
        weights = gmm.weights_[idx_sorted]

        cluster_assignments = gmm.predict(data)
        # 重新映射 cluster id 按 mean 升序
        remap = {old: new for new, old in enumerate(idx_sorted)}
        assignments = np.array([remap[a] for a in cluster_assignments])

        # --- 上: Histogram + GMM 曲線 (TP/FP 疊加) ---
        x_min, x_max = scores_1d.min(), scores_1d.max()
        x_axis = np.linspace(x_min, x_max, 500)
        bin_width = (x_max - x_min) / 40 if x_max > x_min else 1e-6
        scale = len(scores_1d) * bin_width

        # TP / FP 堆疊直方圖
        tp_mask = true_labels == 1
        fp_mask = true_labels == 0
        bins = np.linspace(x_min, x_max, 41)
        ax_hist.hist([scores_1d[tp_mask], scores_1d[fp_mask]], bins=bins,
                     stacked=True, alpha=0.5, color=['steelblue', 'tomato'],
                     label=['TP', 'FP'])

        # GMM 曲線
        for i in range(n_components):
            y = weights[i] * scale * scipy_norm.pdf(x_axis, means[i], stds[i])
            ax_hist.plot(x_axis, y, color=colors[i], linewidth=2, linestyle='--',
                         label=f"C{i} μ={means[i]:.3f}")
            ax_hist.fill_between(x_axis, y, color=colors[i], alpha=0.08)

        ax_hist.set_xlabel(feature_label)
        ax_hist.set_ylabel("Count")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_title(f"1D GMM (k={n_components})\n{feature_label}", fontsize=11)

        # --- 下: 每 cluster 的 TP/FP 堆疊 bar ---
        cluster_ids = list(range(n_components))
        tp_counts = [int(np.sum((assignments == c) & (true_labels == 1))) for c in cluster_ids]
        fp_counts = [int(np.sum((assignments == c) & (true_labels == 0))) for c in cluster_ids]
        totals = [tp + fp for tp, fp in zip(tp_counts, fp_counts)]
        x_pos = np.arange(n_components)

        ax_bar.bar(x_pos, tp_counts, label='TP', color='steelblue',
                   alpha=0.85, edgecolor='black')
        ax_bar.bar(x_pos, fp_counts, bottom=tp_counts, label='FP',
                   color='tomato', alpha=0.85, edgecolor='black')

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([f"C{c}\nμ={means[c]:.3f}" for c in cluster_ids], fontsize=8)
        ax_bar.set_ylabel("Count")
        ax_bar.set_title("各群 TP/FP 組成", fontsize=10)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')

        for c in cluster_ids:
            total = totals[c]
            if total > 0:
                fp_pct = fp_counts[c] / total * 100
                ax_bar.text(c, total + 0.5, f"FP:{fp_pct:.0f}%",
                            ha='center', va='bottom', fontsize=8, color='darkred')

        # 打印摘要 + 回傳 csv rows
        rows = []
        print(f"    {feature_label} 各群組成:")
        for c in cluster_ids:
            t = totals[c]
            fp_pct = fp_counts[c] / t * 100 if t > 0 else 0
            print(f"      Cluster {c} (μ={means[c]:.3f}): "
                  f"TP={tp_counts[c]}, FP={fp_counts[c]}, FP%={fp_pct:.1f}%, n={t}")
            rows.append({
                'feature':    feature_label,
                'cluster_id': c,
                'mean':       round(float(means[c]), 4),
                'weight':     round(float(weights[c]), 4),
                'TP':         tp_counts[c],
                'FP':         fp_counts[c],
                'Total':      t,
                'FP_pct':     round(fp_pct, 2),
                'TP_pct':     round(100 - fp_pct, 2),
            })
        return rows

    # ----- 主繪圖 -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    print(f"\n  [1DGMM] Label {label_name} ({encoder_name}), k={n_components}")
    rows_cd  = _fit_and_plot_1d(axes[0, 0], axes[1, 0], -cd_vals, "-CD (FP嫌疑: 大→高)")
    rows_gap = _fit_and_plot_1d(axes[0, 1], axes[1, 1], gap_vals, "Positive Gap (FP嫌疑: 大→高)")

    # 存圖
    fig.suptitle(
        f"1D GMM 分析  Label: {label_name} [{encoder_name}]\n"
        f"k={n_components}, 總候選樣本={len(cd_vals)} "
        f"(TP={int(true_labels.sum())}, FP={int((true_labels==0).sum())})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    img_fname = f"{save_dir}/1dgmm_{label_name}_{encoder_name}_k{n_components}.png"
    plt.savefig(img_fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1DGMM] Plot  saved: {img_fname}")

    # 存 CSV（CD 和 Gap 兩個特徵的逐群資料合併）
    meta = {
        'label_index': label_index,
        'label_name':  label_name,
        'encoder':     encoder_name,
        'k':           n_components,
        'total_pool':  len(cd_vals),
        'pool_TP':     int(true_labels.sum()),
        'pool_FP':     int((true_labels == 0).sum()),
    }
    all_rows = []
    for r in (rows_cd or []) + (rows_gap or []):
        all_rows.append({**meta, **r})
    if all_rows:
        csv_df   = pd.DataFrame(all_rows)
        csv_fname = f"{save_dir}/1dgmm_{label_name}_{encoder_name}_k{n_components}.csv"
        csv_df.to_csv(csv_fname, index=False)
        print(f"  [1DGMM] Table saved: {csv_fname}")


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
    return df

def visualize_bert_embeddings(model, dataloader, device, num_samples=1000, target_label_index=None, label_names=None):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("正在提取 [CLS] 特徵...")
    with torch.no_grad():
        for batch in dataloader:
            # 取得輸入並搬移到 GPU/CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 你的模型 return (logits, features)
            _, features = model(input_ids, attention_mask)
            
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # 限制採樣數量，避免 t-SNE 跑太久
            if len(np.concatenate(all_embeddings)) >= num_samples:
                break

    # 合併數據
    embeddings = np.concatenate(all_embeddings)[:num_samples]
    labels = np.concatenate(all_labels)[:num_samples]
    
    # 如果指定了特定標籤，則進行過濾或標記
    if target_label_index is not None:
        # 檢查哪些樣本包含目標標籤
        has_target = labels[:, target_label_index] == 1
        
        # 獲取標籤名稱
        label_name = label_names[target_label_index] if label_names is not None else f"Label {target_label_index}"
        
        print(f"\n🎯 目標標籤: {label_name} (index={target_label_index})")
        print(f"   包含此標籤的樣本數: {has_target.sum()} / {len(labels)} ({has_target.sum()/len(labels)*100:.1f}%)")
        
        # 只保留包含目標標籤的樣本
        embeddings_filtered = embeddings[has_target]
        labels_filtered = labels[has_target]
        
        if len(embeddings_filtered) == 0:
            print("❌ 沒有樣本包含此標籤，無法視覺化")
            return
        
        print(f"\n開始計算 t-SNE (處理 {len(embeddings_filtered)} 筆包含 '{label_name}' 的資料)...")
        tsne = TSNE(
            n_components=2, 
            init='pca', 
            learning_rate='auto', 
            perplexity=min(30, len(embeddings_filtered)-1),  # 調整 perplexity
            random_state=42
        )
        low_dim_embs = tsne.fit_transform(embeddings_filtered)
        
        # 建立 DataFrame
        df = pd.DataFrame({
            'tsne_1': low_dim_embs[:, 0],
            'tsne_2': low_dim_embs[:, 1],
            'label_count': labels_filtered.sum(axis=1)
        })
        
        # 繪圖
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            df['tsne_1'], 
            df['tsne_2'], 
            c=df['label_count'], 
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Number of Labels')
        plt.title(f't-SNE Visualization for Label: {label_name}\n(Showing {len(df)} samples, Color = Number of Active Labels)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        output_filename = f'tsne_bert_embeddings_label_{target_label_index}.png'
        plt.savefig(output_filename, dpi=150)
        print(f"✅ t-SNE 視覺化已儲存至 {output_filename}")
        plt.show()
        
    else:
        # 原始的全部標籤視覺化
        # 對於多標籤分類，計算每個樣本的標籤數量
        # labels shape: [num_samples, num_labels]
        label_counts = labels.sum(axis=1)  # 計算每個樣本有多少個標籤
        
        print(f"標籤統計: embeddings shape={embeddings.shape}, labels shape={labels.shape}")
        print(f"標籤數量範圍: {label_counts.min():.0f} - {label_counts.max():.0f}")
        
        # 設定 t-SNE
        print(f"開始計算 t-SNE (處理 {len(embeddings)} 筆資料)...")
        tsne = TSNE(
            n_components=2, 
            init='pca', 
            learning_rate='auto', 
            perplexity=30, 
            random_state=42
        )
        low_dim_embs = tsne.fit_transform(embeddings)
        
        # 建立 DataFrame，使用標籤數量作為顏色
        df = pd.DataFrame({
            'tsne_1': low_dim_embs[:, 0],
            'tsne_2': low_dim_embs[:, 1],
            'label_count': label_counts
        })
        
        # 繪圖
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            df['tsne_1'], 
            df['tsne_2'], 
            c=df['label_count'], 
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Number of Labels')
        plt.title('t-SNE Visualization of BERT [CLS] Embeddings\n(Color = Number of Active Labels)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig('tsne_bert_embeddings.png', dpi=150)
        print(f"✅ t-SNE 視覺化已儲存至 tsne_bert_embeddings.png")
        plt.show()
def visualize_bert_embeddings_multi_labels(model, dataloader, device, target_label_indices, label_names, 
                                            num_samples=1000, custom_labels=None, stage_name=None, 
                                            precomputed_embeddings=None):
    """
    視覺化多個標籤在同一張 t-SNE 圖上
    
    Args:
        model: BERT 模型
        dataloader: 資料載入器
        device: 'cuda' 或 'cpu'
        target_label_indices: 目標標籤索引列表 (例如 [2, 3, 4])
        label_names: 完整的標籤名稱列表
        num_samples: 總採樣數量
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("正在提取 [CLS] 特徵...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            _, features = model(input_ids, attention_mask)
            
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if len(np.concatenate(all_embeddings)) >= num_samples:
                break

    # 合併數據
    embeddings = np.concatenate(all_embeddings)[:num_samples]
    labels = np.concatenate(all_labels)[:num_samples]
    
    # 為每個樣本分配類別標籤
    # 優先順序：如果有多個目標標籤，按照 target_label_indices 的順序分配
    sample_categories = np.full(len(embeddings), -1)  # -1 表示不屬於任何目標標籤
    category_names_map = {}
    
    for idx, label_idx in enumerate(target_label_indices):
        has_label = labels[:, label_idx] == 1
        # 只標記還沒被分類的樣本
        sample_categories[has_label & (sample_categories == -1)] = idx
        category_names_map[idx] = label_names[label_idx]
    
    # 統計每個類別的樣本數
    print(f"\n{'='*70}")
    print("標籤統計:")
    print(f"{'='*70}")
    total_with_target = 0
    for idx, label_idx in enumerate(target_label_indices):
        count = (sample_categories == idx).sum()
        total_with_target += count
        label_name = label_names[label_idx]
        print(f"  {label_name:15s} (index {label_idx:2d}): {count:4d} 樣本 ({count/len(embeddings)*100:5.1f}%)")
    
    no_target_count = (sample_categories == -1).sum()
    print(f"  {'其他標籤':15s}            : {no_target_count:4d} 樣本 ({no_target_count/len(embeddings)*100:5.1f}%)")
    print(f"{'='*70}\n")
    
    # 執行 t-SNE
    print(f"開始計算 t-SNE (處理 {len(embeddings)} 筆資料)...")
    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        perplexity=30,
        random_state=42
    )
    low_dim_embs = tsne.fit_transform(embeddings)
    
    # 繪圖
    plt.figure(figsize=(14, 10))
    
    # 定義顏色方案
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # 先畫其他類別（灰色，透明度低）
    other_mask = sample_categories == -1
    if other_mask.sum() > 0:
        plt.scatter(
            low_dim_embs[other_mask, 0],
            low_dim_embs[other_mask, 1],
            c='lightgray',
            alpha=0.2,
            s=30,
            label='其他標籤',
            edgecolors='none'
        )
    
    # 畫每個目標標籤（顏色鮮明）
    for idx, label_idx in enumerate(target_label_indices):
        mask = sample_categories == idx
        if mask.sum() > 0:
            plt.scatter(
                low_dim_embs[mask, 0],
                low_dim_embs[mask, 1],
                c=colors[idx % len(colors)],
                alpha=0.7,
                s=60,
                label=f"{label_names[label_idx]} ({mask.sum()})",
                edgecolors='white',
                linewidth=0.5
            )
    
    plt.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # 標題
    label_names_str = ', '.join([label_names[i] for i in target_label_indices])
    plt.title(f't-SNE Visualization of BERT Embeddings\nLabels: {label_names_str}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    # 儲存
    output_filename = 'tsne_multi_labels_' + '_'.join([str(i) for i in target_label_indices]) + '.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ t-SNE 視覺化已儲存至 {output_filename}")
    plt.show()
    
def visualize_fp_noise_analysis(embeddings, y_true, y_noisy, y_corrected, label_names, output_filename):
    """視覺化 FP 雜訊分析 - 修正後的邏輯版本"""
    print(f"\n{'='*70}")
    print("FP 雜訊分析（詳細分類）執行中...")
    print(f"{'='*70}")
    
    # 取得索引
    cs_ai_idx = label_names.index('cs.AI')
    cs_lg_idx = label_names.index('cs.LG')
    stat_ml_idx = label_names.index('stat.ML')
    
    # === 核心邏輯運算 ===
    # 1. 基礎定義
    is_fp = (y_true == 0) & (y_noisy == 1)
    is_tp = (y_true == 1) & (y_noisy == 1)
    
    # 2. cs.AI 專屬分類
    cs_ai_fp_mask = is_fp[:, cs_ai_idx]
    cs_ai_fix_fp = cs_ai_fp_mask & (y_corrected[:, cs_ai_idx] == 0)  # 成功訂正
    cs_ai_miss_fp = cs_ai_fp_mask & (y_corrected[:, cs_ai_idx] == 1) # 漏掉沒改 (Miss)
    cs_ai_broke_tp = is_tp[:, cs_ai_idx] & (y_corrected[:, cs_ai_idx] == 0) # 誤殺 TP
    
    # 3. 樣本群體分類 (用於統計)
    has_cs_ai_fp = cs_ai_fp_mask
    is_fp_other = is_fp.copy()
    is_fp_other[:, cs_ai_idx] = False
    has_other_fp = is_fp_other.sum(axis=1) > 0
    
    has_only_other_fp = has_other_fp & ~has_cs_ai_fp
    has_no_fp = ~(has_cs_ai_fp | has_other_fp)
    
    # === 統計輸出 (保持原有的詳盡格式) ===
    print(f"\n【樣本分類統計】")
    print(f"  含 cs.AI FP:     {has_cs_ai_fp.sum():4d} ({has_cs_ai_fp.sum()/len(embeddings)*100:5.1f}%)")
    print(f"  含其他 FP (非AI):{has_only_other_fp.sum():4d} ({has_only_other_fp.sum()/len(embeddings)*100:5.1f}%)")
    print(f"  無 FP 雜訊:      {has_no_fp.sum():4d} ({has_no_fp.sum()/len(embeddings)*100:5.1f}%)")
    
    # === t-SNE 運算 ===
    print("\n正在計算 t-SNE 降維...")
    perplexity_value = min(30, max(5, len(embeddings) - 1))
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perplexity_value, random_state=42)
    low_dim_embs = tsne.fit_transform(embeddings)
    
    # === 繪圖流程 ===
    plt.figure(figsize=(16, 12))
    
    # 定義群組遮罩
    noisy_cs_ai = (y_noisy[:, cs_ai_idx] == 1)
    noisy_cs_lg = (y_noisy[:, cs_lg_idx] == 1)
    noisy_stat_ml = (y_noisy[:, stat_ml_idx] == 1)
    others = ~(noisy_cs_ai | noisy_cs_lg | noisy_stat_ml)
    
    # Layer 1: 背景 (其他)
    plt.scatter(low_dim_embs[others, 0], low_dim_embs[others, 1],
               c='#e8e8e8', alpha=0.2, s=25, label=f'Others ({others.sum()})', zorder=1)
    
    # Layer 2: cs.LG (藍色)
    plt.scatter(low_dim_embs[noisy_cs_lg, 0], low_dim_embs[noisy_cs_lg, 1],
               c='#3498db', alpha=0.6, s=60, label=f'cs.LG ({noisy_cs_lg.sum()})',
               edgecolors='white', linewidth=0.5, zorder=2)
    
    # Layer 3: stat.ML (綠色)
    plt.scatter(low_dim_embs[noisy_stat_ml, 0], low_dim_embs[noisy_stat_ml, 1],
               c='#2ecc71', alpha=0.6, s=60, label=f'stat.ML ({noisy_stat_ml.sum()})',
               edgecolors='white', linewidth=0.5, zorder=3)
    
    # Layer 4: cs.AI (橘色 - 排除掉 Miss_FP 以免重疊)
    cs_ai_display = noisy_cs_ai & ~cs_ai_miss_fp
    plt.scatter(low_dim_embs[cs_ai_display, 0], low_dim_embs[cs_ai_display, 1],
               c='#ff8c42', alpha=0.8, s=70, label=f'cs.AI Normal ({cs_ai_display.sum()})',
               edgecolors='white', linewidth=0.5, zorder=4)
    
    # Layer 5: cs.AI Miss_FP (紅色 X - 重點標註)
    if cs_ai_miss_fp.any():
        plt.scatter(low_dim_embs[cs_ai_miss_fp, 0], low_dim_embs[cs_ai_miss_fp, 1],
                   c='#e74c3c', alpha=1.0, s=150, label=f'cs.AI Miss_FP ({cs_ai_miss_fp.sum()})',
                   marker='X', edgecolors='darkred', linewidth=1.5, zorder=5)

    # 圖表裝飾
    plt.title(f'Noise Analysis: cs.AI Label Correction Status\n(Target: {output_filename})', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', shadow=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_2d_gmm_candidates(cd_vals, gap_vals, prob_clean, label_index,
                                  save_dir="gmm_debug_plots", label_index_path=None,
                                  encoder_name="", true_labels=None, noisy_labels=None):
    """
    繪製 Stage 2 的 2D GMM 分佈圖（CD vs Positive Gap）。

    Args:
        cd_vals:      [n] 候選樣本的 CD 分數
        gap_vals:     [n] 候選樣本的 Positive Gap 分數
        prob_clean:   [n] GMM 輸出的 clean 機率
        label_index:  int，目前的 label index
        save_dir:     圖片儲存目錄
        label_index_path: label 名稱 json 路徑（可選）
        encoder_name: encoder 名稱
        true_labels:  [n] 候選樣本的真實標籤（0 or 1），用於標記真實 TP/FP
        noisy_labels: [n] 候選樣本的 noisy 標籤（候選池都是 1）
    """
    import json
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
    matplotlib.rcParams['axes.unicode_minus'] = False

    if len(cd_vals) < 4:
        return

    os.makedirs(save_dir, exist_ok=True)

    # 取得 label 名稱
    label_name = str(label_index)
    if label_index_path and os.path.exists(label_index_path):
        with open(label_index_path) as f:
            idx2name = {v: k for k, v in json.load(f).items()}
        label_name = idx2name.get(label_index, str(label_index))

    x1 = -cd_vals    # 翻轉 CD：大 = FP 嫌疑高
    x2 =  gap_vals   # Gap：大 = FP 嫌疑高

    has_truth = (true_labels is not None)
    if has_truth:
        true_labels  = np.asarray(true_labels)
        tp_mask = (true_labels == 1)   # noisy=1, true=1 → 真正的 TP
        fp_mask = (true_labels == 0)   # noisy=1, true=0 → 真正的 FP
        n_tp = tp_mask.sum()
        n_fp = fp_mask.sum()

    # ── 圖片版面：有 true_labels 就畫三欄，否則兩欄 ───────────
    ncols = 3 if has_truth else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))

    # ── 欄 0：真實 TP/FP 分佈（觀察 overlap）──────────────────
    if has_truth:
        axes[0].scatter(x1[fp_mask], x2[fp_mask], c='red',   alpha=0.5, s=25,
                        label=f'真實 FP (n={n_fp})', edgecolors='none')
        axes[0].scatter(x1[tp_mask], x2[tp_mask], c='blue',  alpha=0.5, s=25,
                        label=f'真實 TP (n={n_tp})', edgecolors='none')
        axes[0].set_xlabel('-CD  (大 = 遠離原型 = FP嫌疑高)')
        axes[0].set_ylabel('Positive Gap  (大 = FP嫌疑高)')
        axes[0].set_title(f'真實標籤分佈\n（觀察 TP/FP Overlap）')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, linestyle='--', alpha=0.3)
        scatter_ax = axes[1]
        hist_ax    = axes[2]
    else:
        scatter_ax = axes[0]
        hist_ax    = axes[1]

    # ── GMM prob_clean scatter ──────────────────────────────────
    sc = scatter_ax.scatter(x1, x2, c=prob_clean, cmap='RdYlGn',
                            vmin=0, vmax=1, alpha=0.7, edgecolors='gray',
                            linewidth=0.3, s=30)
    plt.colorbar(sc, ax=scatter_ax, label='prob_clean (1=TP, 0=FP)')
    scatter_ax.set_xlabel('-CD  (大 = 遠離原型 = FP嫌疑高)')
    scatter_ax.set_ylabel('Positive Gap  (大 = FP嫌疑高)')
    scatter_ax.set_title(f'Label [{label_index}] {label_name}\nGMM 分群結果 (n={len(cd_vals)})')
    scatter_ax.grid(True, linestyle='--', alpha=0.3)

    # ── prob_clean 直方圖 ───────────────────────────────────────
    if has_truth:
        hist_ax.hist(prob_clean[tp_mask], bins=25, color='blue', alpha=0.5,
                     label=f'真實 TP (n={n_tp})', edgecolor='none')
        hist_ax.hist(prob_clean[fp_mask], bins=25, color='red',  alpha=0.5,
                     label=f'真實 FP (n={n_fp})', edgecolor='none')
        hist_ax.legend(fontsize=9)
    else:
        hist_ax.hist(prob_clean, bins=30, color='steelblue', edgecolor='white', alpha=0.8)

    hist_ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='threshold=0.5')
    hist_ax.set_xlabel('prob_clean')
    hist_ax.set_ylabel('Count')
    hist_ax.set_title('prob_clean 分佈\n（藍=TP, 紅=FP）' if has_truth else 'Distribution of prob_clean')
    hist_ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle(f'TwoStage 2D GMM — Label {label_index} ({label_name})', fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = f"twostage_2dgmm_label{label_index}_{encoder_name}.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ 2D GMM 視覺化已儲存至: {fpath}")   
