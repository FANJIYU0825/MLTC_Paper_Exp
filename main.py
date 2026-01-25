from transformers import BertTokenizer, BertForSequenceClassification,  get_linear_schedule_with_warmup
from torch import optim  as opt
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

# 我的main model
from Rek.solid_gmm_enhance import *
from Rek.experence import NoiseCorrectionEvaluator, ResultRecorder
from util.model import  Mltc
from util.train import train_bert_model,train_once
from util.dataset import create_sample_dataset,DictDataset,load_data_from_tsv,DatasetReassembler
from util.noise_gen import generate_label_dependent_noise

from Rek.experence import get_noise_confusion_matrix
from util.logger import logger
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
    theta: float = 3.0
    alpha: float = 0.7
    epsilon: float = 0.0                     # 會在迴圈中被修改
    num_sample: int = 200                    # 用於 create_sample_dataset (目前沒用到)
    learning_rate: float = 5e-6              # 補上 train_bert_model 可能需要的參數
    epochs: int = 3  
    Noise_type= 'ALL'                     # 訓練輪數
    Resutl_dir: str = './result/'          # 結果儲存目錄
    Noise_ratio = 0.2
# 檔案路徑
FILE_PATH_TRAIN = './dataset/AAPD/train.tsv'
FILE_PATH_VAL = './dataset/AAPD/validation.tsv'
FILE_PATH_TEST = './dataset/AAPD/test.tsv'


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
    
    documents_train, y_true_train, label_names = load_data_from_tsv(FILE_PATH_TRAIN)
    documents_val, y_true_val, _ = load_data_from_tsv(FILE_PATH_VAL) # 不需要重複讀取 label_names
    documents_test, y_true_test, _ = load_data_from_tsv(FILE_PATH_TEST) # 不需要重複讀取 label_names

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

    
    args.Resutl_dir = f'./result_only{args.Noise_type}_ep0/'
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

    # 建立 Datasets 和 DataLoaders
    datasets_train = DictDataset(encoded_batch_train, y_noisy)
    data_loader_train = DataLoader(datasets_train, batch_size=args.batch_size, shuffle=False) # HSL 似乎需要 False
    
    # 測試集使用真實標籤 y_true_test
    datasets_test = DictDataset(encoded_batch_test, y_true_test) 
    data_loader_test = DataLoader(datasets_test, batch_size=args.batch_size, shuffle=False)
    
    return (data_loader_train, data_loader_test, encoded_batch_train, 
            y_true, y_noisy, num_labels, num_samples,label_names,noise_mask)

def get_or_train_warmup_model(args: Args, num_labels: int, data_loader: DataLoader, model_path: str):
    """載入或訓練暖身模型"""
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
    setup_environment(args)
    print('Noise type:',args.Noise_type)
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
        if args.Noise_type == 'ALL':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_noise_0.2_epoch_{epo}.bin'
        elif args.Noise_type == 'FN':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_noise_0.2_fn_epoch_{epo}.bin'
        elif args.Noise_type == 'FP':
            MODEL_PATH_WARMUP = f'{model_path}warm_model_noise_0.2_fp_epoch_{epo}.bin'
        train_once(Mltc,args,epoch_list,num_labels, data_loader_train,MODEL_PATH_WARMUP)
        args.epochs = epo
        
        warmup_model = get_or_train_warmup_model(args, num_labels, data_loader_train, MODEL_PATH_WARMUP)
        
        # 建立 Pipeline 元件
        gmm_filter = GMMNoiseFilter()
        rel_calc = RankWeightedLossCalculator(theta=args.theta)
        cd_calc = PrototypeDistanceCalculator(args)
        
        hsm_pipeline = HSMHybridPipeline(
            rel_calculator=rel_calc,
            cd_calculator=cd_calc,
            filter_strategy=gmm_filter,
            alpha=args.alpha
        )
        
        # 5. 執行 Pipeline 獲取校正結果
        hsm_scores, original_labels, indices = hsm_pipeline.run_score_only(
                model=warmup_model,
                dataloader=data_loader_train,
                device=args.device
                )
        for eps in LS_EP:
            # 建立結果儲存目錄
            if args.Noise_type == 'ALL':
                Result_dir = f'./result_only_ep{eps}/'
            elif args.Noise_type == 'FN':
                Result_dir = f'./result_only_FN{eps}/'
            elif args.Noise_type == 'FP':
                Result_dir = f'./result_only_epFP{eps}/'
            
            os.makedirs(Result_dir, exist_ok=True)

            recorder = ResultRecorder(result_dir=Result_dir)
            logger.info(f'-- Epsilon: {eps} --')
            
            
            args.epsilon = eps
            corrected_labels_global = gmm_filter.correction(hsm_scores, original_labels, args)
            corrected_labels_perlabel = gmm_filter.correction_perlabel(hsm_scores, original_labels, args)

            y_true_aligned = y_true_t[indices]
            y_noisy_aligned = y_noisy_t[indices]
    
            # # 6. 計算統計數據 (重構核心：迴圈邏輯大幅簡化)
            # recorder.clear_records() # 每個 epoch 清空一次紀錄緩衝區
            for i in range(num_labels):
                # 計算 Global 校正數據
                stats_global = evaluator.compute_label_stats(
                    label_index=i, 
                    y_true=y_true_aligned,    # <--- 改用這個
                    y_noisy=y_noisy_aligned,  # <--- 改用這個
                    y_corrected=corrected_labels_global, 
                    method_name='global',
                    args=args
                )
                recorder.add_record(stats_global)

                # Per-Label
                stats_pl = evaluator.compute_label_stats(
                    label_index=i, 
                    y_true=y_true_aligned,    # <--- 改用這個
                    y_noisy=y_noisy_aligned,  # <--- 改用這個
                    y_corrected=corrected_labels_perlabel, 
                    method_name='per_label',
                    args=args
                )
                recorder.add_record(stats_pl)

            # # 7. 存檔
            
            save_filename = f'hsm_gmm_correction_stats_Epoch{epo}_Epsilon{eps}_alp{args.alpha}.csv'
        
            saved_path = recorder.save_to_csv(save_filename)
            logger.info(f"Saved stats for epoch {epo} to {saved_path}")
            
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
        

if __name__ == "__main__":
    # Noise type 可選 'ALL', 'FN', 'FP'
    alpha_list = [0,0.3,0.7,1]
    for alp in alpha_list:
        for noise_type in ['FP']:
    
            Args.Noise_type = noise_type
            Args.alpha=alp
            main_by_epoch()