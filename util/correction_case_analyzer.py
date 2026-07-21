import pandas as pd
import numpy as np
import os
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

def to_numpy(data):
    """將數據轉換為 numpy array，保持原始維度"""
    # 如果是 PyTorch tensor
    if hasattr(data, 'cpu'): 
        return data.detach().cpu().numpy()
    
    # 如果已經是 numpy array，直接返回
    if isinstance(data, np.ndarray):
        return data
    
    # 如果是 list，轉換為 numpy array
    if isinstance(data, list): 
        return np.array(data)
    
    # 其他情況，嘗試轉換（但可能會有維度問題）
    result = np.array(data)
    
    # 檢查是否意外變成 0 維陣列
    if result.ndim == 0:
        raise ValueError(f"轉換後的陣列是 0 維的，原始資料類型: {type(data)}, 值: {data}")
    
    return result


def analyze_cooccurrence_error(
    y_true,
    y_noisy,
    # y_corrected_global,
    y_corrected_perlabel,
    label_names: List[str],
    output_dir: str,
    theta: float,
    epsilon: float = 0.1,
    documents: Optional[List[str]] = None,
    method: str = 'perlabel',
    alpha : float = 0.7,
) -> Dict[str, Any]:
    """
    生成完整的案例級別分析資料庫 (cases.csv)
    
    為每個樣本的每個標籤生成一筆記錄，包含：
    - sample_idx: 樣本索引
    - label_name: 標籤名稱
    - T: Ground Truth (0 或 1)
    - N: Noisy Label (0 或 1)
    - G: Corrected Label (0 或 1, 使用 global correction)
    - G_soft: Corrected Probability (soft值)
    - text: 文檔內容
    
    Args:
        y_true: Ground truth labels (N_samples x N_labels)
        y_noisy: Noisy labels (N_samples x N_labels)
        y_corrected_global: Global corrected labels (N_samples x N_labels)
        y_corrected_perlabel: Per-label corrected labels (N_samples x N_labels)
        label_names: List of label names
        output_dir: Directory to save output files
        epsilon: Threshold for hard/soft classification
        documents: Optional list of document texts
    
    Returns:
        dict: Statistics summary including paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📊 生成案例級別分析資料庫...")
    
    # 轉換為 numpy
    y_true_np = to_numpy(y_true)
    y_noisy_np = to_numpy(y_noisy)
    y_corrected_perlabel_np = to_numpy(y_corrected_perlabel)
    
    n_samples, n_labels = y_true_np.shape
    
    # 準備案例列表
    cases = []
    
    logger.info(f"   處理 {n_samples} 個樣本 x {n_labels} 個標籤 = {n_samples * n_labels} 筆記錄")
    
    # 對每個樣本的每個標籤生成一筆記錄
    for sample_idx in range(n_samples):
        # 獲取文檔文本
        text = documents[sample_idx] if documents and sample_idx < len(documents) else ""
        
        for label_idx, label_name in enumerate(label_names):
            # 獲取該標籤的各階段值
            T = int(y_true_np[sample_idx, label_idx])
            N = int(y_noisy_np[sample_idx, label_idx])
            
            # Global correction
            # G_soft_global = float(y_corrected_global_np[sample_idx, label_idx])
            # G_hard_global = 1 if G_soft_global >= 0.5 else 0
            
            # Per-label correction (保存但不在主要欄位中使用)
            C_soft_perlabel = float(y_corrected_perlabel_np[sample_idx, label_idx])
            if C_soft_perlabel >= 0.5 + epsilon:
                C_hard_perlabel = 1
            elif C_soft_perlabel <= 0.5 - epsilon:
                C_hard_perlabel = 0
            else:
                C_hard_perlabel = C_soft_perlabel  # 模糊地帶，保留原始小數
            # 創建案例記錄 (使用 Global 作為主要結果)
            case = {
                'sample_idx': sample_idx,
                'label_name': label_name,
                'T': T,
                'N': N,
                'C': C_hard_perlabel,
                'soft_C': C_soft_perlabel,
                # 'C_soft': C_soft_perlabel,
                'text': text
            }
            
            cases.append(case)
    
    # 轉換為 DataFrame
    df_cases = pd.DataFrame(cases)
    
    # 儲存到 CSV
    cases_csv_path = os.path.join(output_dir, f'{method}_{theta}_{alpha}_cases.csv')
    df_cases.to_csv(cases_csv_path, index=False)
    logger.info(f"✅ 案例資料庫已儲存: {cases_csv_path}")
    logger.info(f"   總記錄數: {len(df_cases)}")
    
    # 生成統計摘要
    summary_stats = _generate_summary_stats(df_cases, label_names, output_dir, method)
    
    return {
        'cases_csv': cases_csv_path,
        'total_records': len(df_cases),
        'summary': summary_stats
    }


def _generate_summary_stats(df_cases: pd.DataFrame, label_names: List[str], output_dir: str, method: str = '') -> pd.DataFrame:
    """
    生成統計摘要
    
    計算每個標籤的：
    - False Positive 數量
    - False Negative 數量
    - Fix 數量
    - Miss 數量
    """
    summary_records = []
    
    for label_name in label_names:
        # 篩選該標籤的所有記錄
        label_data = df_cases[df_cases['label_name'] == label_name]
        
        # 計算各種情況
        # False Positive: T=0, N=1
        fp_cases = label_data[(label_data['T'] == 0) & (label_data['N'] == 1)]
        fp_fixed = fp_cases[fp_cases['C'] == 0]  # FP 被成功修正
        fp_missed = fp_cases[fp_cases['C'] == 1]  # FP 未被修正
        
        # False Negative: T=1, N=0
        fn_cases = label_data[(label_data['T'] == 1) & (label_data['N'] == 0)]
        fn_fixed = fn_cases[fn_cases['C'] == 1]  # FN 被成功修正
        fn_missed = fn_cases[fn_cases['C'] == 0]  # FN 未被修正
        
        # True Positive: T=1, N=1
        tp_cases = label_data[(label_data['T'] == 1) & (label_data['N'] == 1)]
        tp_kept = tp_cases[tp_cases['C'] == 1]   # TP 保持正確
        tp_broke = tp_cases[tp_cases['C'] == 0]  # TP 被錯誤移除
        
        # True Negative: T=0, N=0
        tn_cases = label_data[(label_data['T'] == 0) & (label_data['N'] == 0)]
        tn_kept = tn_cases[tn_cases['C'] == 0]   # TN 保持正確
        tn_broke = tn_cases[tn_cases['C'] == 1]  # TN 被錯誤添加
        
        summary_records.append({
            'label_name': label_name,
            'total_samples': len(label_data),
            
            # False Positive
            'FP_total': len(fp_cases),
            'FP_fixed': len(fp_fixed),
            'FP_missed': len(fp_missed),
            'FP_fix_rate': len(fp_fixed) / len(fp_cases) if len(fp_cases) > 0 else 0.0,
            
            # False Negative
            'FN_total': len(fn_cases),
            'FN_fixed': len(fn_fixed),
            'FN_missed': len(fn_missed),
            'FN_fix_rate': len(fn_fixed) / len(fn_cases) if len(fn_cases) > 0 else 0.0,
            
            # True Positive
            'TP_total': len(tp_cases),
            'TP_kept': len(tp_kept),
            'TP_broke': len(tp_broke),
            
            # True Negative
            'TN_total': len(tn_cases),
            'TN_kept': len(tn_kept),
            'TN_broke': len(tn_broke),
        })
    
    # 儲存摘要
    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(output_dir, f'{method}_summary.csv' if method else 'summary.csv')
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"✅ 統計摘要已儲存: {summary_path}")
    
    return df_summary