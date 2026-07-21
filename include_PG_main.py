# pyright: reportPrivateImportUsage=false
"""
main.py
=======
MLTC 雜訊校正實驗主程式（已重構為使用 noise_filter 套件）。

主要差異
--------
- 原本 `from Rek.solid_gmm_enhance import *` → 改為從 noise_filter 套件明確匯入
- 移除對 `MathUtils` 的直接依賴（已包進 noise_filter.normalize）
- 行為與原版完全一致：相同的 NPZ 輸出、相同的 CSV 命名、相同的視覺化
"""

from __future__ import annotations

import argparse
import os
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 新套件：noise_filter
# ─────────────────────────────────────────────────────────────────────────────
from Rek import (
    # Calculators
    RankWeightedLossCalculator,
    PrototypeDistanceCalculator,
    PositiveGapCalculator,
    # Filters
    GMMNoiseFilter,
    # Correctors
    RELOnlyCorrector,
    TwoStageRELFPCorrector,
    FrequencyAware1DGMMCorrector,
    # Pipeline
    HSMHybridPipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# 專案內部依賴
# ─────────────────────────────────────────────────────────────────────────────
from Rek.experence import (
    NoiseCorrectionEvaluator,
    ResultRecorder,
    get_noise_confusion_matrix,
)
from util.model       import Mltc, MltcLWAN, MltcLWAN_PerLabel
from util.train       import train_bert_model, train_once
from util.dataset     import DictDataset, load_data_from_tsv
from util.noise_gen   import generate_label_dependent_noise
from util.correction_case_analyzer import analyze_cooccurrence_error
from util.logger      import logger


# ═════════════════════════════════════════════════════════════════════════════
# 設定
# ═════════════════════════════════════════════════════════════════════════════

class Args:
    """所有實驗參數。"""
    # 模型與輸入
    model_name:   str   = "bert-base-uncased"
    max_length:   int   = 512
    batch_size:   int   = 16
    device:       str   = "cuda" if torch.cuda.is_available() else "cpu"
    label_size:   int   = 54
    dropout:      float = 0.1
    seed:         int   = 42

    # HSM / 校正參數
    theta:        float = 3.0
    alpha:        float = 0.5
    beta:         float = 0.5
    epsilon:      float = 0.1

    # 訓練
    learning_rate: float = 5e-6
    epochs:        int   = 3
    num_sample:    int   = 200

    # 雜訊與輸出
    Noise_type:   str   = "FP"
    Noise_ratio:  float = 0.2
    Resutl_dir:   str   = "./result/"

    # Encoder / Normalization / Dataset 選擇
    encoder_name:      str   = "mltc"     # 'mltc' | 'lwan' | 'lwan_perlabel'
    normalization:     str   = "minmax"   # 'minmax' | 'zscore' | 'robust_zscore'
    zscore_clip_range: tuple[float, float] | None = None
    dataset_name:      str   = "AAPD"     # 'AAPD' | 'RCV1'

    # 由 main_by_epoch() 依 dataset_name 動態注入
    
    label_index_path:  str   = ""

    # 小樣本實驗：>0 表示啟用 subsample；同 (dataset, n, seed) 會快取索引
    subsample_n:       int   = 0

    # 視覺化目標 label
    targert_list = [0, 17, 53]


DATASET_PATHS = {
    "AAPD": {
        "train":            "./dataset/AAPD/train.tsv",
        "val":              "./dataset/AAPD/validation.tsv",
        "test":             "./dataset/AAPD/test.tsv",
        "label_index_path": "./dataset/AAPD/label_to_index.json",
    },
    "RCV1": {
        "train":            "./dataset/RCV1/train.tsv",
        "val":              "./dataset/RCV1/validation.tsv",
        "test":             "./dataset/RCV1/test.tsv",
        "label_index_path": "./dataset/RCV1/data/label_to_index.json",
    },
}

ENCODER_MAP = {
    "mltc":          Mltc,
    "lwan":          MltcLWAN,
    "lwan_perlabel": MltcLWAN_PerLabel,
}

# TwoStageRELFPCorrector ablation 配置
TWO_STAGE_CONFIGS = {
    "twostage_rel1pct_2dgmm":   {"top_ratio": 0.01, "n_components": 2},
    "twostage_rel1pct_3comp":   {"top_ratio": 0.01, "n_components": 3},
    "twostage_rel1pct_4comp":   {"top_ratio": 0.01, "n_components": 4},
    "twostage_rel1pct_5comp":   {"top_ratio": 0.01, "n_components": 5},
    "twostage_1dgmm_cd":        {"top_ratio": 0.01, "n_components": 2, "feature_mode": "cd"},
    "twostage_1dgmm_gap":       {"top_ratio": 0.01, "n_components": 2, "feature_mode": "gap"},
    "twostage_1dgmm_cd_3comp":  {"top_ratio": 0.01, "n_components": 3, "feature_mode": "cd"},
    "twostage_1dgmm_gap_3comp": {"top_ratio": 0.01, "n_components": 3, "feature_mode": "gap"},
    "twostage_1dgmm_cd_4comp":  {"top_ratio": 0.01, "n_components": 4, "feature_mode": "cd"},
    "twostage_1dgmm_gap_4comp": {"top_ratio": 0.01, "n_components": 4, "feature_mode": "gap"},
    "twostage_1dgmm_cd_5comp":  {"top_ratio": 0.01, "n_components": 5, "feature_mode": "cd"},
    "twostage_1dgmm_gap_5comp": {"top_ratio": 0.01, "n_components": 5, "feature_mode": "gap"},
}


# ═════════════════════════════════════════════════════════════════════════════
# 環境 / 工具
# ═════════════════════════════════════════════════════════════════════════════

def setup_environment(args: Args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)


# ─────────────────────────────────────────────────────────────────────────────
# Subsample / Noise 快取
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_DIR = "./cache/"


def _get_cache_dir() -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def load_or_make_subsample_indices(args: Args, n_total: int):
    """回傳排序後的 subset indices；subsample_n<=0 或 >=n_total 時回 None。"""
    if args.subsample_n <= 0 or args.subsample_n >= n_total:
        return None
    path = os.path.join(
        _get_cache_dir(),
        f"{args.dataset_name.lower()}_subsample_n{args.subsample_n}_seed{args.seed}.npz",
    )
    if os.path.exists(path):
        idx = np.load(path)["indices"]
        logger.info(f"[cache] Loaded subsample indices: {path} (n={len(idx)})")
        return idx
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_total, size=args.subsample_n, replace=False))
    np.savez(path, indices=idx)
    logger.info(f"[cache] Saved subsample indices: {path}")
    return idx


def load_or_make_noisy_labels(args: Args, y_true_np: np.ndarray) -> np.ndarray:
    """雜訊標籤快取；key = (dataset, subsample_n, seed, noise_type, rho)。"""
    tag_n = args.subsample_n if args.subsample_n > 0 else "full"
    path = os.path.join(
        _get_cache_dir(),
        f"{args.dataset_name.lower()}_n{tag_n}_seed{args.seed}"
        f"_{args.Noise_type}_rho{args.Noise_ratio}.npz",
    )
    if os.path.exists(path):
        y_noisy_np = np.load(path)["y_noisy"]
        logger.info(f"[cache] Loaded noisy labels: {path}")
        return y_noisy_np

    np.random.seed(args.seed)  # 確保 noise_gen 內部的 binomial 可重現
    y_noisy_np = generate_label_dependent_noise(
        y_true=y_true_np, rho=args.Noise_ratio, noise_type=args.Noise_type,
    )
    np.savez(path, y_noisy=y_noisy_np)
    logger.info(f"[cache] Saved noisy labels: {path}")
    return y_noisy_np


def log_initial_noise_stats(y_true, y_noisy, num_labels, label_names, results_dir):
    """記錄校正前的雜訊統計。"""
    logger.info("Logging initial noise statistics...")
    cm_before_all = get_noise_confusion_matrix(y_true=y_true, y_noisy=y_noisy)
    logger.info(f"All labels combined - CM before correction:\n{cm_before_all}")

    output_path = os.path.join(results_dir, "correction_info_before.txt")
    with open(output_path, "w") as f:
        f.write(f"Confusion Matrix before correction (All labels flattened):\n{cm_before_all}\n")
        f.write("=" * 30 + "\n")

    num_samples = y_true.shape[0]
    for i in range(num_labels):
        true_positive_ratio  = y_true[:, i].sum().item() / num_samples
        noisy_positive_count = y_noisy[:, i].sum().item()
        true_positive_count  = y_true[:, i].sum().item()
        cm_label             = get_noise_confusion_matrix(
            y_true=y_true[:, i], y_noisy=y_noisy[:, i]
        )

        logger.info(f"Label {i} ({label_names[i]}):")
        logger.info(f"  True Positive Ratio: {true_positive_ratio:.4f}")
        logger.info(f"  Noisy Positive Count: {noisy_positive_count} (True: {true_positive_count})")
        logger.info(f"  CM:\n{cm_label}")

        with open(output_path, "a") as f:
            f.write(f"Label {i} ({label_names[i]}) Stats:\n")
            f.write(f"  True Positive Ratio: {true_positive_ratio:.4f}\n")
            f.write(f"  Noisy Positive Count: {noisy_positive_count} (True: {true_positive_count})\n")
            f.write(f"  Confusion Matrix:\n{cm_label}\n")
            f.write("--" * 10 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# 資料載入
# ═════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_data(args: Args, tokenizer: BertTokenizer):
    logger.info("Loading and preprocessing data...")

    paths            = DATASET_PATHS[args.dataset_name]
    label_index_path = paths["label_index_path"]
    docs_train, y_train, label_names = load_data_from_tsv(paths["train"], label_index_path=label_index_path)
    docs_val,   y_val,   _           = load_data_from_tsv(paths["val"],   label_index_path=label_index_path)
    docs_test,  y_test,  _           = load_data_from_tsv(paths["test"],  label_index_path=label_index_path)

    # 合併 train + val
    documents_tv = docs_train + docs_val
    y_true_tv    = np.vstack((y_train, y_val))

    # ── 小樣本：用快取的固定 indices 抽 subset ──────────────────────────────
    subset_idx = load_or_make_subsample_indices(args, n_total=len(documents_tv))
    if subset_idx is not None:
        documents_tv = [documents_tv[i] for i in subset_idx]
        y_true_tv    = y_true_tv[subset_idx]
        logger.info(f"[subsample] using {len(subset_idx)} / {subset_idx[-1] + 1}+ samples")

    total_labels = np.sum(y_true_tv)
    print(f"--- 合併後 (Train + Val) ---")
    print(f"樣本總數: {len(documents_tv)}")
    print(f"標籤出現總數: {total_labels}")
    print(f"矩陣形狀: {y_true_tv.shape}")
    print(f"\n--- 測試集 (Test) ---")
    print(f"標籤出現總數: {np.sum(y_test)}")
    print(f"平均每篇文章的標籤數: {total_labels / len(documents_tv):.2f}")

    num_samples     = len(documents_tv)
    num_labels      = len(label_names)
    args.label_size = num_labels

    y_true      = torch.tensor(y_true_tv, dtype=torch.float32)
    y_true_test = torch.tensor(y_test,    dtype=torch.float32)

    # 產生雜訊（含快取）
    logger.info(f"Generating label-dependent noise with rho={args.Noise_ratio}...")
    print("NOISE_RHO", args.Noise_ratio)
    if args.Noise_ratio > 0:
        y_noisy_np = load_or_make_noisy_labels(args, y_true_tv)
        y_noisy = torch.tensor(y_noisy_np, dtype=torch.float32)
    else:
        y_noisy = y_true.clone()

    diff       = np.abs(y_true.numpy() - y_noisy.numpy())
    noise_mask = np.any(diff > 0, axis=1)

    sub_tag = f"_n{args.subsample_n}" if args.subsample_n > 0 else ""
    args.Resutl_dir = (
        f"./result_{args.dataset_name}_{args.encoder_name}{sub_tag}_only{args.Noise_type}"
        f"_ep{args.epsilon}_theta{args.theta}_alpha{args.alpha}/"
    )
    os.makedirs(args.Resutl_dir, exist_ok=True)
    log_initial_noise_stats(y_true, y_noisy, num_labels, label_names, args.Resutl_dir)

    # Tokenization
    logger.info("Tokenizing datasets...")
    enc_train = tokenizer.batch_encode_plus(
        documents_tv, add_special_tokens=True, padding="max_length",
        truncation=True, max_length=args.max_length,
        return_attention_mask=True, return_tensors="pt",
    )
    enc_test = tokenizer.batch_encode_plus(
        docs_test, add_special_tokens=True, padding="max_length",
        truncation=True, max_length=args.max_length,
        return_attention_mask=True, return_tensors="pt",
    )

    ds_train = DictDataset(enc_train, y_noisy,      texts=documents_tv)
    ds_test  = DictDataset(enc_test,  y_true_test,  texts=docs_test)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False)
    loader_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False)

    return (loader_train, loader_test, enc_train,
            y_true, y_noisy, num_labels, num_samples, label_names, noise_mask)


# ═════════════════════════════════════════════════════════════════════════════
# 模型準備
# ═════════════════════════════════════════════════════════════════════════════

def get_or_train_warmup_model(
    args:        Args,
    num_labels:  int,
    data_loader: DataLoader,
    model_path:  str,
):
    """載入或訓練暖身模型（支援三種 encoder）。"""
    ModelClass = ENCODER_MAP[args.encoder_name]
    logger.info(f"Using {ModelClass.__name__} encoder")

    warmup_model = ModelClass(num_labels)

    if os.path.exists(model_path):
        logger.info(f"Loading existing warmup model from {model_path}")
        warmup_model.load_state_dict(torch.load(model_path, map_location=args.device))
        warmup_model.to(args.device)
    else:
        logger.info(f"Training new warmup model, will save to {model_path}")
        warmup_model.to(args.device)
        warmup_model = train_bert_model(
            warmup_model, data_loader,
            epochs=args.epochs, device=args.device, warmup=True,
        )
        torch.save(warmup_model.state_dict(), model_path)
        logger.info(f"Warmup model saved to {model_path}")

    return warmup_model


def build_warmup_model_path(args: Args, epoch: int) -> str:
    """根據實驗設定組合 warmup 模型的存檔路徑。"""
    os.makedirs("./model/", exist_ok=True)
    ds_suffix  = args.dataset_name.lower()
    enc_suffix = args.encoder_name
    noise_tag  = {"ALL": "_all", "FN": "_fn", "FP": "_fp"}[args.Noise_type]
    sub_tag    = f"_n{args.subsample_n}" if args.subsample_n > 0 else ""
    return (
        f"./model/warm_model_{ds_suffix}_{enc_suffix}{sub_tag}"
        f"_noise_{args.Noise_ratio}{noise_tag}_epoch_{epoch}.bin"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 校正流程
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_scores(
    pipeline:     HSMHybridPipeline,
    model,
    data_loader:  DataLoader,
    args:         Args,
):
    """
    跑一次模型推論，取得 raw / norm 分數（避免重複計算）。

    Returns
    -------
    rel_raw, cd_raw, gap_raw, rel_norm, cd_norm, gap_norm,
    original_labels, indices, hsm_scores
    """
    (rel_raw, cd_raw, gap_raw,
     rel_norm, cd_norm, gap_norm,
     original_labels, indices) = pipeline.run_score_separately(
        model=model,
        dataloader=data_loader,
        device=args.device,
        encoder_name=args.encoder_name,
        normalization=args.normalization,
        clip_range=args.zscore_clip_range,
        dataset_name=args.dataset_name.lower(),
    )

    # 從 normalized 分數推導融合的 HSM 分數
    cd_part    = args.beta * cd_norm + (1 - args.beta) * gap_norm
    hsm_scores = args.alpha * rel_norm + (1 - args.alpha) * cd_part

    return (rel_raw, cd_raw, gap_raw,
            rel_norm, cd_norm, gap_norm,
            original_labels, indices, hsm_scores)


def save_scores_npz(
    result_dir: str,
    args:       Args,
    rel_raw, cd_raw, gap_raw,
    rel_norm, cd_norm, gap_norm,
    labels, indices,
) -> None:
    """存分數 NPZ（若檔案不存在）。"""
    sub_tag = f"_n{args.subsample_n}" if args.subsample_n > 0 else ""
    path = os.path.join(
        result_dir,
        f"{args.dataset_name}_{args.encoder_name}{sub_tag}_{args.Noise_type}"
        f"_{args.theta}_{args.epsilon}_{args.alpha}.npz",
    )
    if os.path.exists(path):
        return
    np.savez_compressed(
        path,
        rel_scores=rel_raw,    rel_norm=rel_norm,
        cd_scores=cd_raw,      cd_norm=cd_norm,
        margin_scores=gap_raw, margin_norm=gap_norm,
        labels=labels, indices=indices,
        alpha=args.alpha, beta=args.beta,
        normalization=args.normalization,
    )
    logger.info(f"Scores NPZ saved: {path}")


def run_all_correctors(
    rel_raw, cd_raw, gap_raw,
    hsm_scores,
    original_labels,
    y_true_aligned,
    args:        Args,
    result_dir:  str,
    eps:         float,
) -> dict:
    """執行所有校正方法，回傳 {method_name: corrected_labels}。"""
    all_corrected = {}

    # NOTE: 舊 HSM `double_gmm_per_label` baseline 已停用
    # （Rek 套件重構後 GMMNoiseFilter 介面變更，且 CLAUDE.md 標記為 ablation-only）

    # ── 1. RELOnlyCorrector（Stage 1 only 基線）─────────────────────────────
    for ratio in (0.01, 0.05):
        key = f"stage1_rel_only_{int(ratio * 100)}pct"
        all_corrected[key] = RELOnlyCorrector(top_ratio=ratio).correct(
            rel_scores=rel_raw, labels=original_labels,
        )

    # ── 2. TwoStageRELFPCorrector ablation ─────────────────────────────────
    for method_name, params in TWO_STAGE_CONFIGS.items():
        corrector = TwoStageRELFPCorrector(**params)
        all_corrected[method_name] = corrector.correct(
            rel_scores=rel_raw,
            cd_scores=cd_raw,
            gap_scores=gap_raw,
            labels=original_labels,
            args=args,
            y_true=y_true_aligned.numpy(),
        )

    # ── 3. FrequencyAware 1D GMM ───────────────────────────────────────────
    fa1d = FrequencyAware1DGMMCorrector(
        n_components=2,
        head_epsilon=eps,
        mid_epsilon=eps,
        tail_epsilon=min(eps * 2.0, 0.2),
    )

    # 4a. 分布分析（CSV）
    fa1d.analyze(
        cd_raw, gap_raw, original_labels,
        save_path=os.path.join(result_dir, f"fa1dgmm_analyze_ep{eps}.csv"),
    )

    # 4b. 一次性校正
    all_corrected["fa1d_gmm"] = fa1d.correct(cd_raw, gap_raw, original_labels)

    # 4c. 漸進式校正 → CSV
    _, prog_stats = fa1d.progressive_correct(
        cd_raw, gap_raw, original_labels,
        y_true=y_true_aligned.numpy(),
        n_rounds=20,
    )
    prog_path = os.path.join(result_dir, f"fa1dgmm_progressive_ep{eps}.csv")
    pd.DataFrame(prog_stats).to_csv(prog_path, index=False)
    logger.info(f"[FA1DGMM] Progressive stats saved → {prog_path}")

    return all_corrected


def evaluate_and_record(
    all_corrected:   dict,
    y_true_aligned,
    y_noisy_aligned,
    num_labels:      int,
    label_names,
    evaluator:       NoiseCorrectionEvaluator,
    recorder:        ResultRecorder,
    args:            Args,
    data_loader:     DataLoader,
    epoch:           int,
    eps:             float,
) -> None:
    """逐 label 計算統計、執行案例分析、存 CSV。"""
    # 1. 統計
    for i in range(num_labels):
        for method_name, corrected_labels in all_corrected.items():
            stats = evaluator.compute_label_stats(
                label_index=i,
                y_true=y_true_aligned,
                y_noisy=y_noisy_aligned,
                y_corrected=corrected_labels,
                method_name=method_name,
                args=args,
            )
            recorder.add_record(stats)

    # 2. 案例分析
    try:
        documents = getattr(data_loader.dataset, "texts", None)
        if documents is None:
            logger.warning("DataLoader 未包含文本數據，使用空字串代替")
            documents = [""] * len(y_true_aligned)

        for method_name, corrected_labels in all_corrected.items():
            analyze_cooccurrence_error(
                y_true=y_true_aligned,
                y_noisy=y_noisy_aligned,
                y_corrected_perlabel=corrected_labels,
                label_names=label_names,
                output_dir=args.Resutl_dir,
                theta=args.theta,
                epsilon=args.epsilon,
                alpha=args.alpha,
                documents=documents,
                method=method_name,
            )
    except Exception as e:
        logger.error(f"案例分析失敗: {e}")
        logger.error(traceback.format_exc())

    # 3. 存 CSV
    save_filename = (
        f"hsm_gmm_stats_al{args.alpha}_Ep{epoch}_Eps{eps}"
        f"_theta{args.theta}_gmm_comparing.csv"
    )
    try:
        saved_path = recorder.save_to_csv(save_filename)
        print(f"✅ CSV 成功寫入: {saved_path}")
        if not os.path.exists(saved_path):
            print(f"❌ 警告: 函式回傳成功但找不到檔案")
    except Exception as e:
        print(f"❌ 寫入 CSV 時發生錯誤: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════

def main_by_epoch():
    # 1. 初始化
    args                  = Args()
    args.label_index_path = DATASET_PATHS[args.dataset_name]["label_index_path"]
    setup_environment(args)
    print("Noise type:",   args.Noise_type)
    print(f"Encoder type: {args.encoder_name}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 2. 資料載入
    (data_loader_train, data_loader_test, _,
     y_true, y_noisy, num_labels, num_samples, label_names, _,
     ) = load_and_preprocess_data(args, tokenizer)

    y_true_t  = y_true  if isinstance(y_true,  torch.Tensor) else torch.tensor(y_true)
    y_noisy_t = y_noisy if isinstance(y_noisy, torch.Tensor) else torch.tensor(y_noisy)

    # 3. 評估器
    evaluator = NoiseCorrectionEvaluator(label_names)

    # 4. 各 epoch
    epoch_list = [args.epochs]
    LS_EP      = [0.05]

    for epo in epoch_list:
        logger.info(f"--- Starting Epoch: {epo} ---")
        args.epochs = epo

        # 4a. 暖身模型（含預訓練 baseline）
        warmup_path = build_warmup_model_path(args, epo)
        ModelClass  = ENCODER_MAP[args.encoder_name]
        train_once(ModelClass, args, epoch_list, num_labels, data_loader_train, warmup_path)
        warmup_model = get_or_train_warmup_model(
            args, num_labels, data_loader_train, warmup_path,
        )

        # 4b. 建立分數計算 pipeline
        _sub  = f"_n{args.subsample_n}" if args.subsample_n > 0 else ""
        file_prefix = (
            f"{args.dataset_name.lower()}_{args.encoder_name}"
            f"{_sub}_{args.Noise_type}"
        )
        hsm_pipeline = HSMHybridPipeline(
            rel_calculator=RankWeightedLossCalculator(
                theta=args.theta, file_prefix=file_prefix,
            ),
            cd_calculator=PrototypeDistanceCalculator(args),
            gap_calculator=PositiveGapCalculator(),
            alpha=args.alpha,
            beta=args.beta,
        )

        # 4c. 一次計算所有分數
        (rel_raw, cd_raw, gap_raw,
         rel_norm, cd_norm, gap_norm,
         original_labels, indices, hsm_scores) = compute_all_scores(
            hsm_pipeline, warmup_model, data_loader_train, args,
        )

        # 4d. 各 epsilon 設定
        for eps in LS_EP:
            logger.info(f"-- Epsilon: {eps} --")
            args.epsilon = eps

            sub_tag = f"_n{args.subsample_n}" if args.subsample_n > 0 else ""
            result_dir = (
                f"./result_{args.dataset_name}_{args.encoder_name}{sub_tag}"
                f"_only{args.Noise_type}_ep{eps}_theta{args.theta}_alpha{args.alpha}/"
            )
            os.makedirs(result_dir, exist_ok=True)

            save_scores_npz(
                result_dir, args,
                rel_raw, cd_raw, gap_raw,
                rel_norm, cd_norm, gap_norm,
                original_labels, indices,
            )

            recorder        = ResultRecorder(result_dir=result_dir)
            y_true_aligned  = y_true_t[indices]
            y_noisy_aligned = y_noisy_t[indices]

            # 跑所有校正方法
            all_corrected = run_all_correctors(
                rel_raw, cd_raw, gap_raw, hsm_scores,
                original_labels, y_true_aligned,
                args, result_dir, eps,
            )

            # 評估 + 記錄 + 存 CSV
            evaluate_and_record(
                all_corrected, y_true_aligned, y_noisy_aligned,
                num_labels, label_names,
                evaluator, recorder, args,
                data_loader_train, epo, eps,
            )


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Run MLTC Experiments")
    parser.add_argument("--encode",        type=str,   default="lwan",
                        choices=["mltc", "lwan", "lwan_perlabel"])
    parser.add_argument("--noise_type",    type=str,   default="FP",
                        choices=["ALL", "FN", "FP"])
    parser.add_argument("--alpha",         type=float, default=0.7)
    parser.add_argument("--normalization", type=str,   default="minmax",
                        choices=["minmax", "zscore", "robust_zscore"])
    parser.add_argument("--zscore_clip",   type=float, default=None,
                        help="Z-score clipping range (e.g., 5 for [-5, 5])")
    parser.add_argument("--dataset",       type=str,   default="AAPD",
                        choices=["AAPD", "RCV1"])
    parser.add_argument("--subsample",     type=int,   default=0,
                        help="若 >0 則固定取 N 筆子集（同 seed 會快取索引重用）")
    parser.add_argument("--epochs",        type=int,   default=3,
                        help="Warmup epochs")
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_args()

    Args.Noise_type       = cmd_args.noise_type
    Args.encoder_name     = cmd_args.encode
    Args.dataset_name     = cmd_args.dataset
    Args.normalization    = cmd_args.normalization
    Args.subsample_n      = cmd_args.subsample
    Args.epochs           = cmd_args.epochs
    Args.zscore_clip_range = (
        (-cmd_args.zscore_clip, cmd_args.zscore_clip)
        if cmd_args.zscore_clip is not None else None
    )

    for alpha_val in [cmd_args.alpha]:
        print(f"\n{'=' * 60}")
        print("  Running experiment:")
        print(f"    Encoder:       {Args.encoder_name}")
        print(f"    Alpha:         {alpha_val}")
        print(f"    Noise Type:    {Args.Noise_type}")
        print(f"    Normalization: {Args.normalization}")
        if Args.zscore_clip_range:
            print(f"    Z-Score Clip:  {Args.zscore_clip_range}")
        print(f"{'=' * 60}\n")

        Args.alpha = alpha_val
        main_by_epoch()