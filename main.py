from transformers import BertTokenizer, BertForSequenceClassification,  get_linear_schedule_with_warmup
from torch import optim  as opt
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

# 我的main model
from Rek.solid_gmm_enhance import *
from Rek.experence import NoiseCorrectionEvaluator, ResultRecorder,get_noise_confusion_matrix,visualize_bert_embeddings
from util.model import Mltc, MltcLWAN, MltcLWAN_PerLabel  # 新增三種 encoder
from util.train import train_bert_model,train_once
from util.dataset import create_sample_dataset,DictDataset,load_data_from_tsv,DatasetReassembler
from util.noise_gen import generate_label_dependent_noise
from util.correction_case_analyzer import  analyze_cooccurrence_error
from util.logger import logger
# from util.correction_case_analyzer import analyze_corrections
import os
# 其他標準庫
import numpy as np
import copy
import gc


class Args:
    """存放所有實驗參數"""
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model_name: str = "bert-base-uncased"   #
    max_length: int = 64                   # BERT 輸入的最大長度 (原 MAX_LEN)
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    label_size: int = 54                     # 多標籤的類別數 (會被資~料集覆蓋)
    dropout: float = 0.1
    seed: int = 42
    theta: float = 3.0                      # RankWeightedLossCalculator 參數
    alpha: float = 0.7                      # HSMHybridPipeline 參數：REL 佔比
    beta: float = 0.5                       # HSMHybridPipeline 參數：CD 在 (1-alpha) 中的佔比（beta=1.0 退化為原始 HSM）
    epsilon: float = 0.05                     # 會在迴圈中被修改
    num_sample: int = 200                    # 用於 create_sample_dataset (目前沒用到)
    learning_rate: float = 5e-6              # 補上 train_bert_model 可能需要的參數
    epochs: int = 3
    Noise_type= 'FP'                     # 訓練輪數
    Resutl_dir: str = './result/'          # 結果儲存目錄
    Noise_ratio = 0.2

    # Encoder 類型選擇 (新增)
    # 選項: 'mltc' (Baseline), 'lwan' (Proposed), 'lwan_perlabel' (Advanced)
    encoder_name: str = 'mltc'              # 預設使用 baseline

    # Normalization 標準化方法選擇 (新增)
    # 選項: 'minmax' (原始), 'zscore' (Z-Score), 'robust_zscore' (穩健 Z-Score)
    normalization: str = 'minmax'           # 預設使用原始的 min-max
    zscore_clip_range: tuple = None         # Z-Score 裁剪範圍，例如 (-5, 5) 或 None

    # Dataset 選擇: 'AAPD' (54 labels) | 'RCV1' (103 labels)
    dataset_name: str = 'AAPD'

    # 原有類別 (已生成 GMM plots): [0,1,3,4,5,6,8,9,11,12,13,14, 2, 7, 10, 15, 17, 19]
    # 測試 GMM 3-component 繪圖功能: 只使用前 5 個標籤
    targert_list = [0, 1, 2, 3, 4]  # 測試用,只繪製前 5 個標籤
    # targert_list = [i for i in range(54)]   # AAPD 完整版本
    # targert_list = [i for i in range(103)]  # RCV1 完整版本

# 檔案路徑 (由 dataset_name 決定，在 load_and_preprocess_data 中設定)
DATASET_PATHS = {
    'AAPD': {
        'train':            './dataset/AAPD/train.tsv',
        'val':              './dataset/AAPD/validation.tsv',
        'test':             './dataset/AAPD/test.tsv',
        'label_index_path': './dataset/AAPD/label_to_index.json',
    },
    'RCV1': {
        'train':            './dataset/RCV1/train.tsv',
        'val':              './dataset/RCV1/validation.tsv',
        'test':             './dataset/RCV1/test.tsv',
        'label_index_path': './dataset/RCV1/data/label_to_index.json',
    },
}


def setup_environment(args: Args):
    """設定隨機種子和 device"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
def log_initial_noise_stats(y_true, y_noisy, num_labels, label_names, results_dir):
    """記錄並儲存校正前的雜訊統計數據"""
    logger.info("Logging initial noise statistics...")
    
    # 計算全域混淆矩陣
    cm_before_all = get_noise_confusion_matrix(y_true=y_true, y_noisy=y_noisy)
    logger.info(f"All labels combined - CM before correction:\n{cm_before_all}")
    
    output_path = os.path.join(results_dir, 'correction_info_before.txt')
    with open(output_path, 'w') as f:
        f.write(f"Confusion Matrix before correction (All labels flattened):\n{cm_before_all}\n")
        f.write("="*30 + "\n")

    # 計算並記錄每個標籤的混淆矩陣
    num_samples = y_true.shape[0]
    for i in range(num_labels):
        true_positive_ratio = y_true[:, i].sum().item() / num_samples
        noisy_positive_count = (y_noisy[:, i]).sum().item()
        true_positive_count = y_true[:, i].sum().item()
        cm_label = get_noise_confusion_matrix(y_true=y_true[:, i], y_noisy=y_noisy[:, i])
        
        logger.info(f"Label {i} ({label_names[i]}):")
        logger.info(f"  True Positive Ratio: {true_positive_ratio:.4f}")
        logger.info(f"  Noisy Positive Count: {noisy_positive_count} (True: {true_positive_count})")
        logger.info(f"  CM:\n{cm_label}")
        
        with open(output_path, 'a') as f:
            f.write(f"Label {i} ({label_names[i]}) Stats:\n")
            f.write(f"  True Positive Ratio: {true_positive_ratio:.4f}\n")
            f.write(f"  Noisy Positive Count: {noisy_positive_count} (True: {true_positive_count})\n")
            f.write(f"  Confusion Matrix:\n{cm_label}\n")
            f.write("--"*10 + "\n")   
def load_and_preprocess_data(args: Args, tokenizer: BertTokenizer):
    """載入、預處理、產生雜訊並建立 DataLoaders"""
    logger.info("Loading and preprocessing data...")
    
    paths = DATASET_PATHS[args.dataset_name]
    label_index_path = paths['label_index_path']
    documents_train, y_true_train, label_names = load_data_from_tsv(paths['train'], label_index_path=label_index_path)
    documents_val, y_true_val, _ = load_data_from_tsv(paths['val'], label_index_path=label_index_path)
    documents_test, y_true_test, _ = load_data_from_tsv(paths['test'], label_index_path=label_index_path)

    # 2. 合併數據 (Train + Val + Test)
    # 如果您只想算 Train + Val，就把 Test 的部分拿掉
    documents_train_val = documents_train + documents_val
    y_true_train_val = np.vstack((y_true_train, y_true_val))
    

    total_labels_train_val = np.sum(y_true_train_val)

    # 4. 輸出結果
    print(f"--- 合併後 (Train + Val) ---")
    print(f"樣本總數: {len(documents_train_val)}")
    print(f"標籤出現總數 (Total Label Occurrences): {total_labels_train_val}")
    print(f"矩陣形狀: {y_true_train_val.shape}") # 確認一下形狀是否正確
    average_labels = total_labels_train_val / len(documents_train_val)
    print(f"\n--- 測試集 (Test) ---")
    print(f"標籤出現總數: {np.sum(y_true_test)}")
    print(f"平均每篇文章的標籤數: {average_labels:.2f}")
    
    # (可選) 縮減資料集規模以便S debug
    # documents, y_true = documents[:100], y_true[:100]
    
    num_samples = len(documents_train_val)
    num_labels = len(label_names)
    args.label_size = num_labels # 更新 args 中的 label_size

    # 轉換為 Tensor
    y_true = torch.tensor(y_true_train_val, dtype=torch.float32)
    y_true_test = torch.tensor(y_true_test, dtype=torch.float32)

    # 產生雜訊
    logger.info(f"Generating label-dependent noise with rho={args.Noise_ratio}...")
    noise_rho=args.Noise_ratio
    print('NOISE_RHO',noise_rho)
    
    if noise_rho > 0:
        # 呼叫您的雜訊生成函數
        # 注意：您的函數回傳 numpy，這裡轉回 Tensor
        y_noisy_np = generate_label_dependent_noise(
            y_true=y_true_train_val, # 傳入 numpy 版本
            rho=noise_rho,
            noise_type=args.Noise_type
        )
        y_noisy = torch.tensor(y_noisy_np, dtype=torch.float32)
    else:
        # 如果 rho=0，雜訊標籤 = 乾淨標籤
        y_noisy = y_true.clone()

    diff = np.abs(y_true.numpy() - y_noisy.numpy())
    noise_mask = np.any(diff > 0, axis=1)
    # 記錄雜訊統計

    
    args.Resutl_dir = f'./result_{args.dataset_name}_{args.encoder_name}_only{args.Noise_type}_ep{args.epsilon}_theta{args.theta}_alpha{args.alpha}/'
    os.makedirs(args.Resutl_dir, exist_ok=True)
    log_initial_noise_stats(y_true, y_noisy, num_labels, label_names, args.Resutl_dir)

    # Tokenization
    logger.info("Tokenizing datasets...")
    encoded_batch_train = tokenizer.batch_encode_plus(
        documents_train_val,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=args.max_length,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    encoded_batch_test = tokenizer.batch_encode_plus(
        documents_test,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=args.max_length,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # 建立 Datasets 和 DataLoaders (傳入原始文本)
    datasets_train = DictDataset(encoded_batch_train, y_noisy, texts=documents_train_val)
    data_loader_train = DataLoader(datasets_train, batch_size=args.batch_size, shuffle=False) # HSL 似乎需要 False
    
    # 測試集使用真實標籤 y_true_test
    datasets_test = DictDataset(encoded_batch_test, y_true_test, texts=documents_test) 
    data_loader_test = DataLoader(datasets_test, batch_size=args.batch_size, shuffle=False)
    
    return (data_loader_train, data_loader_test, encoded_batch_train, 
            y_true, y_noisy, num_labels, num_samples,label_names,noise_mask)

def get_or_train_warmup_model(args: Args, num_labels: int, data_loader: DataLoader, model_path: str):
    """載入或訓練暖身模型 - 支援三種 encoder"""

    # 根據 args.encoder_name 選擇模型
    if args.encoder_name == 'lwan':
        logger.info(f"Using MltcLWAN encoder (Shared W)")
        warmup_model = MltcLWAN(num_labels)
    elif args.encoder_name == 'lwan_perlabel':
        logger.info(f"Using MltcLWAN_PerLabel encoder (Per-label W)")
        warmup_model = MltcLWAN_PerLabel(num_labels)
    else:  # 'mltc' or default
        logger.info(f"Using Mltc encoder (Baseline - CLS token)")
        warmup_model = Mltc(num_labels)

    if os.path.exists(model_path):
        logger.info(f"Loading existing warmup model from {model_path}")
        warmup_model.load_state_dict(torch.load(model_path, map_location=args.device))
        warmup_model.to(args.device)
    else:
        logger.info(f"Training new warmup model, will save to {model_path}")
        warmup_model.to(args.device)
        warmup_model = train_bert_model(warmup_model, data_loader, epochs=args.epochs, device=args.device,warmup=True)
        torch.save(warmup_model.state_dict(), model_path)
        logger.info(f"Warmup model saved to {model_path}")

    return warmup_model
def main_by_epoch():
    # 1. 初始化與環境設定
    args = Args()
    args.label_index_path = DATASET_PATHS[args.dataset_name]['label_index_path']
    # alpha 由 Args class 或 __main__ 中的 Args.alpha 控制，不在此覆蓋
    setup_environment(args)
    print('Noise type:',args.Noise_type)
    print(f'Encoder type: {args.encoder_name}')  # 顯示使用的 encoder
    # tokenizer = BertTokenizer.from_pretrained(args.model_name)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 2. 資料載入
    (data_loader_train, data_loader_test, encoded_batch_train,
     y_true, y_noisy, num_labels, num_samples, label_names,noise_mask) = load_and_preprocess_data(args, tokenizer)

    # 預先轉為 Tensor，避免在迴圈中重複轉換 (效能優化)
    y_true_t = torch.tensor(y_true) if isinstance(y_true, np.ndarray) else y_true
    y_noisy_t = torch.tensor(y_noisy) if isinstance(y_noisy, np.ndarray) else torch.tensor(y_noisy)

    # 3. 初始化輔助類別
    evaluator = NoiseCorrectionEvaluator(label_names)


    epoch_list = [3]
    LS_EP = [0.05]
    for epo in epoch_list:
        logger.info(f'--- Starting Epoch: {epo} ---')
        # 4. 準備模型與 Pipeline
        # 路徑不存在
        model_path = './model/'
        os.makedirs(model_path , exist_ok=True)
        # 根據 encoder 類型決定模型路徑 (支援比較實驗)
        encoder_suffix = args.encoder_name  # 使用 encoder_name

        # 根據 encoder_name 選擇模型類別
        if args.encoder_name == 'lwan':
            ModelClass = MltcLWAN
        elif args.encoder_name == 'lwan_perlabel':
            ModelClass = MltcLWAN_PerLabel
        else:
            ModelClass = Mltc

        dataset_suffix = args.dataset_name.lower()
        if args.Noise_type == 'ALL':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_{dataset_suffix}_{encoder_suffix}_noise_{args.Noise_ratio}_epoch_{epo}.bin'
        elif args.Noise_type == 'FN':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_{dataset_suffix}_{encoder_suffix}_noise_{args.Noise_ratio}_fn_epoch_{epo}.bin'
        elif args.Noise_type == 'FP':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_{dataset_suffix}_{encoder_suffix}_noise_{args.Noise_ratio}_fp_epoch_{epo}.bin'

        train_once(ModelClass, args, epoch_list, num_labels, data_loader_train, MODEL_PATH_WARMUP)
        args.epochs = epo
        
        warmup_model = get_or_train_warmup_model(args, num_labels, data_loader_train, MODEL_PATH_WARMUP)
        
        # 建立 Pipeline 元件
        gmm_filter = GMMNoiseFilter()
        dataset_suffix = args.dataset_name.lower()
        encoder_suffix = args.encoder_name
        file_prefix = f"{dataset_suffix}_{encoder_suffix}"
        rel_calc = RankWeightedLossCalculator(theta=args.theta, file_prefix=file_prefix)
        cd_calc = PrototypeDistanceCalculator(args)
        gap_calc = PositiveGapCalculator()
        hsm_pipeline = HSMHybridPipeline(
            rel_calculator=rel_calc,
            cd_calculator=cd_calc,
            alpha=args.alpha,
            gap_calculator=gap_calc,
            beta=args.beta,
        )
        triple_gmm_refiner = LabelRefiner(n_components=3) # 或者是 2
        quad_gmm_refiner   = LabelRefiner(n_components=4)
        penta_gmm_refiner  = LabelRefiner(n_components=5)
        # 5. 執行 Pipeline 獲取校正結果 負責組合 RankWeightedLoss 和 PrototypeDistance
        hsm_scores, original_labels, indices = hsm_pipeline.run_score_only(
                model=warmup_model,
                dataloader=data_loader_train,
                device=args.device,
                encoder_name=encoder_suffix,
                normalization=args.normalization,
                clip_range=args.zscore_clip_range,
                dataset_name=dataset_suffix,
                )
        for eps in LS_EP:
            # 建立結果儲存目錄 (加入 encoder 名稱)
            encoder_suffix = getattr(args, 'encoder_name', 'mltc')
            Result_dir = f'./result_{args.dataset_name}_{encoder_suffix}_only{args.Noise_type}_ep{eps}_theta{args.theta}_alpha{args.alpha}/'

            os.makedirs(Result_dir, exist_ok=True)

            recorder = ResultRecorder(result_dir=Result_dir)
            logger.info(f'-- Epsilon: {eps} --')
            
            
            args.epsilon = eps
            # corrected_labels_global = gmm_filter.correction(hsm_scores, original_labels, args)
            double_gmm_corrected_labels_perlabel = gmm_filter.correction_perlabel(hsm_scores, original_labels, args)
            triple_gmmcorrected_labels_perlabel=triple_gmm_refiner.refine(hsm_scores, original_labels,args)
            quad_gmmcorrected_labels_perlabel=quad_gmm_refiner.refine(hsm_scores, original_labels,args)
            penta_gmmcorrected_labels_perlabel=penta_gmm_refiner.refine(hsm_scores, original_labels,args)
            y_true_aligned = y_true_t[indices]
            y_noisy_aligned = y_noisy_t[indices]
    
            # # 6. 計算統計數據 (重構核心：迴圈邏輯大幅簡化)
            # recorder.clear_records() # 每個 epoch 清空一次紀錄緩衝區
            for i in range(num_labels):
                # 計算 Global 校正數據
                # stats_global = evaluator.compute_label_stats(
                #     label_index=i, 
                #     y_true=y_true_aligned,    # <--- 改用這個
                #     y_noisy=y_noisy_aligned,  # <--- 改用這個
                #     y_corrected=corrected_labels_global, 
                #     method_name='global',
                #     args=args
                # )
                # recorder.add_record(stats_global)

                # Per-Label
                stats_pl = evaluator.compute_label_stats(
                    label_index=i, 
                    y_true=y_true_aligned,    # <--- 改用這個
                    y_noisy=y_noisy_aligned,  # <--- 改用這個
                    y_corrected=double_gmm_corrected_labels_perlabel, 
                    method_name='double_gmm_per_label',
                    args=args
                )
                recorder.add_record(stats_pl)
                # triple gmm
                stats_pl_triple = evaluator.compute_label_stats(
                    label_index=i, 
                    y_true=y_true_aligned,    # <--- 改用這個
                    y_noisy=y_noisy_aligned,  # <--- 改用這個
                    y_corrected=triple_gmmcorrected_labels_perlabel, 
                    method_name='triple_gmm_per_label',
                    args=args
                )
                recorder.add_record(stats_pl_triple)
                # quad gmm (4-comp)
                stats_pl_quad = evaluator.compute_label_stats(
                    label_index=i, 
                    y_true=y_true_aligned,
                    y_noisy=y_noisy_aligned,
                    y_corrected=quad_gmmcorrected_labels_perlabel, 
                    method_name='quad_gmm_per_label',
                    args=args
                )
                recorder.add_record(stats_pl_quad)
                # penta gmm (5-comp)
                stats_pl_penta = evaluator.compute_label_stats(
                    label_index=i, 
                    y_true=y_true_aligned,
                    y_noisy=y_noisy_aligned,
                    y_corrected=penta_gmmcorrected_labels_perlabel, 
                    method_name='penta_gmm_per_label',
                    args=args
                )
                recorder.add_record(stats_pl_penta)
            # === 新增: 案例級別分析 ===
            # logger.info("開始案例級別分析...")
            try:
                # 提取訓練集文本內容（從 DataLoader 的 dataset）
                documents = data_loader_train.dataset.texts
                
                if documents is None:
                    logger.warning("DataLoader 未包含文本數據，cases.csv 的 text 欄位將為空")
                    documents = [''] * len(y_true_aligned)
                
                case_result_double =  analyze_cooccurrence_error(
                    y_true=y_true_aligned,
                    y_noisy=y_noisy_aligned,
                    # y_corrected_global=corrected_labels_global,
                    y_corrected_perlabel=double_gmm_corrected_labels_perlabel,  # 修正：使用校正後的標籤陣列，而非統計字典
                    label_names=label_names,
                    output_dir=args.Resutl_dir,
                    theta=args.theta,
                    epsilon =args.epsilon,
                    documents=documents , # 傳入訓練集文本
                    alpha=args.alpha,
                    method = 'double_gmm_per_label'
                )
                case_result_triple =  analyze_cooccurrence_error(
                    y_true=y_true_aligned,
                    y_noisy=y_noisy_aligned,
                    y_corrected_perlabel=triple_gmmcorrected_labels_perlabel,  # 修正：使用校正後的標籤陣列，而非統計字典
                    label_names=label_names,
                    output_dir=args.Resutl_dir,
                    theta=args.theta,
                    epsilon =args.epsilon,
                    documents=documents , # 傳入訓練集文本
                    alpha=args.alpha,
                    method = 'triple_gmm_per_label'
                )
                case_result_quad =  analyze_cooccurrence_error(
                    y_true=y_true_aligned,
                    y_noisy=y_noisy_aligned,
                    y_corrected_perlabel=quad_gmmcorrected_labels_perlabel,
                    label_names=label_names,
                    output_dir=args.Resutl_dir,
                    theta=args.theta,
                    epsilon =args.epsilon,
                    alpha=args.alpha,
                    documents=documents,
                    method = 'quad_gmm_per_label'
                )
                case_result_penta =  analyze_cooccurrence_error(
                    y_true=y_true_aligned,
                    y_noisy=y_noisy_aligned,
                    y_corrected_perlabel=penta_gmmcorrected_labels_perlabel,
                    label_names=label_names,
                    output_dir=args.Resutl_dir,
                    theta=args.theta,
                    epsilon =args.epsilon,
                    alpha=args.alpha,
                    documents=documents,
                    method = 'penta_gmm_per_label'
                )
            #     # logger.info(f"✅ 案例分析完成！")
            #     # logger.info(f"   - 案例資料庫: {Result_dir}/cases.csv")
            #     # logger.info(f"   - 統計摘要: {Result_dir}/summary.csv")
            except Exception as e:
                logger.error(f"案例分析失敗: {e}")
                import traceback
                logger.error(traceback.format_exc())

            # # # 7. 存檔
            
            # save_filename = f'hsm_gmm_stats_Ep{epo}_Eps{eps}_alp{args.alpha}_gmmcomparing.csv'
            save_filename = f'hsm_gmm_stats_al{args.alpha}_Ep{epo}_Eps{eps}_theta{args.theta}_gmm_comparing.csv'
            
            try:
                saved_path = recorder.save_to_csv(save_filename)
                print(f"✅ CSV 成功寫入: {saved_path}") # 明確告訴你檔案在哪
                
                # 再次確認檔案是否存在
                if os.path.exists(saved_path):
                    print(f"   (檔案確認存在)")
                else:
                    print(f"❌ 警告: 函式回傳成功但找不到檔案")
                    
            except Exception as e:
                print(f"❌ 寫入 CSV 時發生錯誤: {e}")
            
            # # 8. 建立新的 DataLoader 供後續訓練使用
            # # 確保結果是 Tensor
            # if not isinstance(corrected_labels_global, torch.Tensor):
            #     corrected_labels_global = torch.tensor(corrected_labels_global)
            # if not isinstance(corrected_labels_perlabel, torch.Tensor):
            #     corrected_labels_perlabel = torch.tensor(corrected_labels_perlabel)
            

            # corrected_global_datasloader=DatasetReassembler.create_retraining_loader(
            #     encoded_inputs=encoded_batch_train,
            #     corrected_labels=corrected_labels_global,
            #     batch_size=args.batch_size
            # )
            # corrected_perlabel_datasloader=DatasetReassembler.create_retraining_loader(
            #     encoded_inputs=encoded_batch_train,
            #     corrected_labels=corrected_labels_perlabel,
            #     batch_size=args.batch_size
            # )
            # # retrain 
            # logger.info(f'--- Retraining with Global Corrected Labels for Epoch: {epo} ---')
            # model_for_global = copy.deepcopy(warmup_model)
            # retrain_model_global = train_bert_model(
            #     model_for_global, 
            #     corrected_global_datasloader, 
            #     epochs=args.epochs, 
            #     device=args.device,
            #     warmup=True
            # )
            # # retrain perlabel
            # logger.info(f'--- Retraining with Per-Label Corrected Labels for Epoch: {epo} ---')
            # model_for_perlabel = copy.deepcopy(warmup_model)
            # retrain_model_perlabel = train_bert_model(
            #     model_for_perlabel, 
            #     corrected_perlabel_datasloader, 
            #     epochs=args.epochs, 
            #     device=args.device,
            #     warmup=True
            # )
            # # evaluation
            # result_global =eval_model(retrain_model_global, data_loader_test, args.device)
            # result_perlabel =eval_model(retrain_model_perlabel, data_loader_test, args.device)
            # logger.info(f'--- Evaluation Result after Global Correction Retraining at Epoch {epo} ---')
            # recorder.clear_records()
            # # 1. 取得各類別的 DataFrame
            # df_global= result_global['per_class_df']
            # df_perlabel= result_perlabel['per_class_df']
            # # 2. 依照 F1-score 排序，找出表現最差的 5 個類別
            # print("表現最差的 5 個類別：")
            # print(df_global.sort_values(by='f1-score', ascending=True).head(5))
            # print(df_perlabel.sort_values(by='f1-score', ascending=True).head(5))
            # # 3. 依照 F1-score 排序，找出表現最好的 5 個類別
            # print("\n表現最好的 5 個類別：")
            # print(df_global.sort_values(by='f1-score', ascending=False).head(5))
            # print(df_perlabel.sort_values(by='f1-score', ascending=False).head(5))
            # df_global.to_csv(Result_dir+f'per_class_global_retrain_epoch{epo}_Epsilon{eps}.csv', index=False)
            # df_perlabel.to_csv(Result_dir+f'per_class_perlabel_retrain_epoch{epo}_Epsilon{eps}.csv', index=False)
            # recorder.add_record({
            #     'epoch': epo,
            #     'correction_method': 'global_retrain',
                
            #     'test_accuracy': result_global['accuracy'],
            #     'test_f1_micro': result_global['f1_micro'],
            #     'test_f1_macro': result_global['f1_macro'],
            #     'test_precision':result_global['precision'],
                
            # })
            
            # recorder.add_record({
            #     'epoch': epo,
            #     'correction_method': 'per_label_retrain',
                
            #     'test_accuracy': result_perlabel['accuracy'],
            #     'test_f1_micro': result_perlabel['f1_micro'],
            #     'test_f1_macro': result_perlabel['f1_macro'],
            #     'test_precision': result_perlabel['precision'],
            # })
    
            # # logger.info(f"Saved evaluation results for epoch {epo} to {saved_path}")
            # save_filename = f'evaluation_results_epoch{epo}_Epsilon{eps}.csv'
            # logger.info(f'Saving evaluation results to {save_filename}')
            # saved_path = recorder.save_to_csv(save_filename)
def proto_vis ():
    args = Args()
    setup_environment(args)
    print('Noise type:',args.Noise_type)
    print(f'Encoder type: {args.encoder_name}')  # 顯示使用的 encoder
    # tokenizer = BertTokenizer.from_pretrained(args.model_name)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 2. 資料載入
    (data_loader_train, data_loader_test, encoded_batch_train,
     y_true, y_noisy, num_labels, num_samples, label_names,noise_mask) = load_and_preprocess_data(args, tokenizer)

    # 預先轉為 Tensor，避免在迴圈中重複轉換 (效能優化)
    y_true_t = torch.tensor(y_true) if isinstance(y_true, np.ndarray) else y_true
    y_noisy_t = torch.tensor(y_noisy) if isinstance(y_noisy, np.ndarray) else torch.tensor(y_noisy)

    # 3. 初始化輔助類別
    evaluator = NoiseCorrectionEvaluator(label_names)


    epoch_list = [3]
    LS_EP = [0.05]
    for epo in epoch_list:
        logger.info(f'--- Starting Epoch: {epo} ---')
        # 4. 準備模型與 Pipeline
        # 路徑不存在
        model_path = './model/'
        os.makedirs(model_path , exist_ok=True)
        # 根據 encoder 類型決定模型路徑 (支援比較實驗)
        encoder_suffix = args.encoder_name  # 使用 encoder_name

        # 根據 encoder_name 選擇模型類別
        if args.encoder_name == 'lwan':
            ModelClass = MltcLWAN
        elif args.encoder_name == 'lwan_perlabel':
            ModelClass = MltcLWAN_PerLabel
        else:
            ModelClass = Mltc

        dataset_suffix = args.dataset_name.lower()
        if args.Noise_type == 'ALL':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_{dataset_suffix}_{encoder_suffix}_noise_{args.Noise_ratio}_epoch_{epo}.bin'
        elif args.Noise_type == 'FN':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_{dataset_suffix}_{encoder_suffix}_noise_{args.Noise_ratio}_fn_epoch_{epo}.bin'
        elif args.Noise_type == 'FP':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_{dataset_suffix}_{encoder_suffix}_noise_{args.Noise_ratio}_fp_epoch_{epo}.bin'

        train_once(ModelClass, args, epoch_list, num_labels, data_loader_train, MODEL_PATH_WARMUP)
        args.epochs = epo
        
        warmup_model = get_or_train_warmup_model(args, num_labels, data_loader_train, MODEL_PATH_WARMUP)
        
        # 建立 Pipeline 元件
        # gmm_filter = GMMNoiseFilter()
        # rel_calc = RankWeightedLossCalculator(theta=args.theta)
        # cd_calc = PrototypeDistanceCalculator(args)
        
        # hsm_pipeline = HSMHybridPipeline(
        #     rel_calculator=rel_calc,
        #     cd_calculator=cd_calc,
        #     filter_strategy=gmm_filter,
        #     alpha=args.alpha
        # )
        # # 5. 執行 Pipeline 獲取校正結果
        # hsm_scores, original_labels, indices = hsm_pipeline.run_score_only(
        #         model=warmup_model,
        #         dataloader=data_loader_train,
        #         device=args.device
        #         )
        dataset_suffix = args.dataset_name.lower()
        encoder_suffix = args.encoder_name
        for eps in LS_EP:
            # 建立結果儲存目錄
            Result_dir = f'./result_{args.dataset_name}_{encoder_suffix}_only{args.Noise_type}_ep{eps}_theta{args.theta}_alpha{args.alpha}/'
            
            os.makedirs(Result_dir, exist_ok=True)

            recorder = ResultRecorder(result_dir=Result_dir)
            logger.info(f'-- Epsilon: {eps} --')
            
            
            args.epsilon = eps
            # corrected_labels_global = gmm_filter.correction(hsm_scores, original_labels, args)
            # corrected_labels_perlabel = gmm_filter.correction_perlabel(hsm_scores, original_labels, args)

            # y_true_aligned = y_true_t[indices]
            # y_noisy_aligned = y_noisy_t[indices]
            visualize_bert_embeddings(warmup_model, data_loader_train, device=args.device, num_samples=1000, label_names=label_names)
            
            # 範例：視覺化特定標籤 (例如 cs.AI，索引為 13)
            # visualize_bert_embeddings(warmup_model, data_loader_train, device=args.device, num_samples=1000, target_label_index=13, label_names=label_names)
            
            # # 6. 計算統計數據 (重構核心：迴圈邏輯大幅簡化)
            # recorder.clear_records() # 每個 epoch 清空一次紀錄緩衝區
            # for i in range(num_labels):
            #     # 計算 Global 校正數據
            #     stats_global = evaluator.compute_label_stats(
            #         label_index=i, 
            #         y_true=y_true_aligned,    # <--- 改用這個
            #         y_noisy=y_noisy_aligned,  # <--- 改用這個
            #         y_corrected=corrected_labels_global, 
            #         method_name='global',
            #         args=args
            #     )
            #     recorder.add_record(stats_global)

            #     # Per-Label
            #     stats_pl = evaluator.compute_label_stats(
            #         label_index=i, 
            #         y_true=y_true_aligned,    # <--- 改用這個
            #         y_noisy=y_noisy_aligned,  # <--- 改用這個
            #         y_corrected=corrected_labels_perlabel, 
            #         method_name='per_label',
            #         args=args
            #     )
            #     recorder.add_record(stats_pl)

            # # === 新增: 案例級別分析 ===
            # logger.info("開始案例級別分析...")
            # try:
            #     # 提取訓練集文本內容（從 DataLoader 的 dataset）
            #     documents = data_loader_train.dataset.texts
                
            #     if documents is None:
            #         logger.warning("DataLoader 未包含文本數據，cases.csv 的 text 欄位將為空")
            #         documents = [''] * len(y_true_aligned)
                
            #     case_result =  analyze_cooccurrence_error(
            #         y_true=y_true_aligned,
            #         y_noisy=y_noisy_aligned,
            #         y_corrected_global=corrected_labels_global,
            #         y_corrected_perlabel=corrected_labels_perlabel,
            #         label_names=label_names,
            #         output_dir=Result_dir,
            #         epsilon=args.epsilon,
            #         documents=documents  # 傳入訓練集文本
            #     )
            #     logger.info(f"✅ 案例分析完成！")
            #     logger.info(f"   - 案例資料庫: {Result_dir}/cases.csv")
            #     logger.info(f"   - 統計摘要: {Result_dir}/summary.csv")
            # except Exception as e:
            #     logger.error(f"案例分析失敗: {e}")
            #     import traceback
            #     logger.error(traceback.format_exc())

            # # # 7. 存檔
            
            # save_filename = f'hsm_gmm_stats_Ep{epo}_Eps{eps}_alp{args.alpha}.csv'
            
            # try:
            #     saved_path = recorder.save_to_csv(save_filename)
            #     print(f"✅ CSV 成功寫入: {saved_path}") # 明確告訴你檔案在哪
                
            #     # 再次確認檔案是否存在
            #     if os.path.exists(saved_path):
            #         print(f"   (檔案確認存在)")
            #     else:
            #         print(f"❌ 警告: 函式回傳成功但找不到檔案")
                    
            # except Exception as e:
            #     print(f"❌ 寫入 CSV 時發生錯誤: {e}")
        
        
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run MLTC Experiments')
    parser.add_argument('--encode', type=str, default='lwan_perlabel',
                        choices=['mltc', 'lwan', 'lwan_perlabel'],
                        help='Encoder to use for the experiment')
    parser.add_argument('--noise_type', type=str, default='FP',
                        choices=['ALL', 'FN', 'FP'],
                        help='Noise type')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Alpha value for HSM pipeline')
    parser.add_argument('--normalization', type=str, default='minmax',
                        choices=['minmax', 'zscore', 'robust_zscore'],
                        help='Normalization method for REL and CD scores')
    parser.add_argument('--zscore_clip', type=float, default=None,
                        help='Z-score clipping range (e.g., 5 for [-5, 5]), None for no clipping')
    parser.add_argument('--dataset', type=str, default='AAPD',
                        choices=['AAPD', 'RCV1'],
                        help='Dataset to use for the experiment')

    cmd_args = parser.parse_args()

    # Noise type 可選 'ALL', 'FN', 'FP'
    Args.Noise_type = cmd_args.noise_type

    # ============================================
    # Encoder 選擇
    # ============================================
    # 選項: 'mltc' (Baseline), 'lwan' (Proposed), 'lwan_perlabel' (Advanced)
    Args.encoder_name = cmd_args.encode
    Args.dataset_name = cmd_args.dataset

    # ============================================
    # Normalization 選擇
    # ============================================
    Args.normalization = cmd_args.normalization
    if cmd_args.zscore_clip is not None:
        Args.zscore_clip_range = (-cmd_args.zscore_clip, cmd_args.zscore_clip)
    else:
        Args.zscore_clip_range = None

    # ============================================
    # 實驗執行
    # ============================================
    for alpha_val in [cmd_args.alpha]:  # 根據參數測試 alpha
        print(f"\n{'='*60}")
        print(f"  Running experiment:")
        print(f"    Encoder: {Args.encoder_name}")
        print(f"    Alpha: {alpha_val}")
        print(f"    Noise Type: {Args.Noise_type}")
        print(f"    Normalization: {Args.normalization}")
        if Args.zscore_clip_range:
            print(f"    Z-Score Clip Range: {Args.zscore_clip_range}")
        print(f"{'='*60}\n")
        Args.alpha = alpha_val
        main_by_epoch()