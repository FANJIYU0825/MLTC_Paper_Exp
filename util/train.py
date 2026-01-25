
from torch.optim import AdamW
from tqdm.auto import tqdm # 用於顯示進度條
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

import numpy as np
from sklearn.metrics import f1_score, accuracy_score,classification_report, precision_score, recall_score
import pandas as pd
import seaborn as sns
import json
import torch.nn as nn
import torch
import torch.optim.adamw as Adamw
import torch.nn.functional as F
import os
# myself
from util.logger import logger
from util.model import  Mltc
from util.dataset import create_sample_dataset,DictDataset,load_data_from_tsv,DatasetReassembler

def train_bert_model(model, data_loader, epochs, device,warmup=False):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(data_loader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            b_input_ids= batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            model.zero_grad()
            logits, _ = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            loss = criterion(logits, b_labels)
            total_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()

        avg_train_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.4f}")
        if warmup:
            print("Warmup training log: Checking loss distribution...")
        
        # [修正 2] 傳入當前的 epoch 變數，而不是寫死 0
        # check_gmm_readiness(model, data_loader, device, epoch=epoch+1)
        # dry_run_gmm(model, data_loader, device)

    return model


def train_once(model_class,args, epoch_list: list, num_labels: int, data_loader: DataLoader,MODEL_PATH_WARMUP: str):
    # model_class: 傳入模型類別 (例如 Mltc)
    
    # 1. 初始化模型
    warmup_model = model_class(num_labels)
    warmup_model.to(args.device)
    
    current_trained_epoch = 0 
    
    for target_epo in epoch_list:
        # --- 修正點：檔名必須在這裡動態產生 ---
        # 這裡將 epoch 格式化進去，例如 model_epoch_10.bin, model_epoch_20.bin
        # current_model_path = f'./model/warm_model_noise_0.2_epoch_{target_epo}.bin'

        current_model_path=MODEL_PATH_WARMUP.replace('{epo}', str(target_epo))
        
        print(f'--- Checking Checkpoint for Target Epoch: {target_epo} ---')

        # --- 情況 A: 檔案已經存在 (直接讀取，跳過訓練) ---
        if os.path.exists(current_model_path):
            logger.info(f"Checkpoint found: {current_model_path}. Loading...")
            
            # 讀取檔案
            warmup_model.load_state_dict(torch.load(current_model_path, map_location=args.device))
            
            # 更新目前進度指標
            current_trained_epoch = target_epo
            logger.info(f"Model loaded. Jump to epoch: {current_trained_epoch}")
            
        # --- 情況 B: 檔案不存在 (需要補練) ---
        else:
            # 計算差額
            epochs_needed = target_epo - current_trained_epoch
            
            if epochs_needed > 0:
                logger.info(f"Target file not found. Training from {current_trained_epoch} to {target_epo} ({epochs_needed} epochs)...")
                
                # 接續訓練
                warmup_model = train_bert_model(
                    warmup_model, 
                    data_loader, 
                    epochs=epochs_needed, 
                    device=args.device, 
                    warmup=True
                )
                
                # 訓練完後，存成「這個階段」的檔名
                torch.save(warmup_model.state_dict(), current_model_path)
                logger.info(f"Saved checkpoint: {current_model_path}")
                
                # 更新進度
                current_trained_epoch = target_epo
            else:
                logger.info("Epochs needed <= 0, skipping.")

    return warmup_model

