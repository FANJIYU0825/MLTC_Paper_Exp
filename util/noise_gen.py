import numpy as np
import torch

def _false_negative_noise(label_list, rho_10):
    # np.random.seed(42)
    fn_noise = [0] * len(label_list)
    for i, label in enumerate(label_list):
        if label == 1 and np.random.binomial(1, rho_10) == 1:
            fn_noise[i] = 1 # This vector marks which '1's to remove
    return np.array(fn_noise)


def _false_positive_noise(normalized_co_mat, label_list, rho_01):
        
        # np.random.seed(42)
        fp_noise = [0] * len(label_list)
        if sum(label_list) == 0: # Cannot generate FP noise if no positive labels exist
            return np.array(fp_noise)
        trans_vec = np.zeros(len(label_list))
        for i, label in enumerate(label_list):
            if label == 1:
                trans_vec += normalized_co_mat[i]

        # Normalize transition vector and apply noise rate
        
        # print('Sum of label_list:', sum(label_list))
        # print('trans_vec',trans_vec.shape)
        trans_vec = trans_vec / sum(label_list) * rho_01*100

        # Don't flip existing positive labels
        trans_vec[np.array(label_list) == 1] = 0

        # Generate noise based on the calculated probabilities
        for i, label in enumerate(label_list):
            if label == 0:
                prob = min(trans_vec[i], 1.0) # Ensure prob is not > 1
                if np.random.binomial(1, prob) == 1:
                    fp_noise[i] = 1 # This vector marks which '0's to add
        return np.array(fp_noise)
       
        


def generate_label_dependent_noise( y_true: np.ndarray, rho: float = 0.4,noise_type :str = 'ALL') -> np.ndarray:
    
    # Compute the noise transition vector based on co-occurrence
    

    # --- Main Logic ---
    if hasattr(y_true, 'numpy'):  # Check for PyTorch/TensorFlow Tensors
        y_true = y_true.numpy()
    elif not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    # 1. Calculate co-occurrence matrix from the true labels
    co_mat = np.dot(y_true.T, y_true)
    print('Co-occurrence Matrix:\n', co_mat.shape)
    np.fill_diagonal(co_mat, 0)

    # Add a small epsilon to avoid division by zero for rows that sum to 0
    row_sums = np.sum(co_mat, axis=1)[:, np.newaxis]
    normalized_co_mat = co_mat / (row_sums + 1e-8)

    # 2. Calculate noise rates
    L_avg = y_true.sum(axis=1).mean()
    L = y_true.shape[1]
    
    # 設定理論雜訊率
    rho_10 = rho # False Negative Rate
    rho_01 = rho * L_avg / (L - L_avg) if L > L_avg else rho # False Positive Rate
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    # 2. 以 FP 為主，計算預期的錯誤數量 (Expected Noise Count)
    expected_fp_count = n_neg * rho_01

    # 3. 動態計算 rho_10 (FN rate)，使得預期 FN 數量等於 FP 數量
    if n_pos > 0:
        rho_10 = expected_fp_count / n_pos
    else:
        rho_10 = 0.0

    # [安全機制] 確保 rho_10 不會超過 1.0 (如果正樣本太少，可能無法產生同等數量的噪音)
    rho_10 = np.clip(rho_10, 0.0, 1.0)

    print(f"設定 FP rate (rho_01): {rho_01:.4f} (預期產生 {int(expected_fp_count)} 個 FP)")
    print(f"計算 FN rate (rho_10): {rho_10:.4f} (預期產生 {int(n_pos * rho_10)} 個 FN)")
    # 3. Generate FN and FP noise for each sample
    fn_labels = np.array([_false_negative_noise(y, rho_10) for y in y_true])
    fp_labels = np.array([_false_positive_noise(normalized_co_mat, y, rho_01) for y in y_true])
    
    # 4. Combine clean labels with generated noise
    # Formula: noisy = clean - false_negatives + false_positives
    if noise_type == 'ALL':
        y_noisy = y_true - fn_labels + fp_labels
    elif noise_type == 'FN':
        y_noisy = y_true - fn_labels
    elif noise_type == 'FP':
        y_noisy = y_true + fp_labels
    else:
        # 防呆機制，若無匹配則保持原樣
        y_noisy = y_true.copy()

    # --- Print Summary (移到最後) ---
   # 統計實際發生的變化
    diff = y_noisy - y_true
    
    # 統計數量
    n_fn = np.sum(diff == -1)  # 實際的 False Negatives 數量
    n_fp = np.sum(diff == 1)   # 實際的 False Positives 數量
    n_total_flips = n_fn + n_fp # 總共變動的標籤數
    
    # 統計總標籤數 (用於計算比率)
    total_elements = y_true.size
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)

    print("-" * 30)
    print("NOISE ANALYSIS REPORT")
    print("-" * 30)
    
    # 1. 數量統計
    print(f"Total Flips           : {n_total_flips} / {total_elements} elements")
    print(f"  - False Negatives   : {n_fn} (flipped 1 -> 0)")
    print(f"  - False Positives   : {n_fp} (flipped 0 -> 1)")
    
    # 2. 實際雜訊率 (Actual Noise Rates) vs 設定值
    # 避免分母為 0
    actual_rho_10 = n_fn / total_positives if total_positives > 0 else 0
    actual_rho_01 = n_fp / total_negatives if total_negatives > 0 else 0
    
    print(f"\nActual vs Target Rates:")
    print(f"  - FN Rate (rho_10)  : Actual {actual_rho_10:.4f} vs Target {rho_10:.4f}")
    print(f"  - FP Rate (rho_01)  : Actual {actual_rho_01:.4f} vs Target {rho_01:.4f}")
    # Ensure labels are binary (0 or 1) after operations
    y_noisy = np.clip(y_noisy, 0, 1)
    return y_noisy.astype(float)