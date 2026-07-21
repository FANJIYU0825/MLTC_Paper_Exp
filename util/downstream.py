"""
util/downstream.py
==================
Downstream evaluation utilities for MLTC ablation and epsilon sensitivity studies.

Components
----------
ExperimentLogger  — append-safe master CSV log with deduplication by run_id
run_downstream_eval — retrain from warmup checkpoint on corrected labels, eval on test set

Recording schema (one row = one complete experiment):
    run_id, timestamp, dataset, encoder, noise_type, noise_ratio,
    alpha, beta, epsilon, theta, correction_method, ablation_variant,
    retrain_epochs,
    noisy_f1_micro, noisy_f1_macro,          # model trained on noisy labels
    corrected_f1_micro, corrected_f1_macro,   # model trained on corrected labels
    delta_f1_micro, delta_f1_macro,           # improvement over noisy baseline
    precision, recall, accuracy,
    fix_fp_total, fix_fn_total,               # 8-state summary (summed across labels)
    broke_tp_total, broke_tn_total,
    miss_fp_total, miss_fn_total,
    keep_tp_total, keep_tn_total
"""

import os
import hashlib
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ── optional Weights & Biases logging ───────────────────────────────────────
try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def _wandb_log(metrics: dict) -> None:
    """有 active wandb run 時才記錄；沒裝 wandb 或沒 init 時靜默略過。"""
    if _wandb is not None and getattr(_wandb, 'run', None) is not None:
        _wandb.log(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentLogger
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Append-safe experiment log backed by a single CSV file.

    Usage
    -----
    logger = ExperimentLogger('results/experiment_log.csv')

    config = dict(dataset='AAPD', encoder='lwan', noise_type='FP',
                  noise_ratio=0.2, alpha=0.7, beta=0.5, epsilon=0.05,
                  theta=3.0, correction_method='double_gmm_perlabel',
                  ablation_variant='full', retrain_epochs=5)

    if not logger.is_done(config):
        metrics = run_downstream_eval(...)
        logger.log(config, metrics)
    """

    COLUMNS = [
        'run_id', 'timestamp',
        # ── experiment config ──
        'dataset', 'encoder', 'noise_type', 'noise_ratio',
        'alpha', 'beta', 'epsilon', 'theta',
        'correction_method', 'ablation_variant', 'retrain_epochs',
        'init_mode',  # 'warmup' = 接續 warmup checkpoint, 'scratch' = 從頭 finetune
        'model_path',  # retrain 完的模型存檔路徑
        # ── noisy-label baseline (warmup model on noisy data) ──
        'noisy_f1_micro', 'noisy_f1_macro',
        # ── downstream result ──
        'corrected_f1_micro', 'corrected_f1_macro',
        'delta_f1_micro', 'delta_f1_macro',
        'precision', 'recall', 'accuracy',
        # ── correction quality (8-state, summed across labels) ──
        'fix_fp_total', 'fix_fn_total',
        'broke_tp_total', 'broke_tn_total',
        'miss_fp_total', 'miss_fn_total',
        'keep_tp_total', 'keep_tn_total',
    ]

    def __init__(self, log_path: str = 'results/experiment_log.csv'):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

        if os.path.exists(log_path):
            self.df = pd.read_csv(log_path)
            # ensure all columns exist (backward compat when schema grows)
            for col in self.COLUMNS:
                if col not in self.df.columns:
                    self.df[col] = np.nan
        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)

    # ── identity ──────────────────────────────────────────────────────────────

    @staticmethod
    def make_run_id(config: dict) -> str:
        """Deterministic 10-char hex ID from config keys that identify a run."""
        id_keys = ['dataset', 'encoder', 'noise_type', 'noise_ratio',
                   'alpha', 'beta', 'epsilon', 'theta',
                   'correction_method', 'ablation_variant', 'retrain_epochs',
                   'init_mode']
        sub = {k: config[k] for k in id_keys if k in config}
        key = json.dumps(sub, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:10]

    def is_done(self, config: dict) -> bool:
        run_id = self.make_run_id(config)
        return run_id in self.df['run_id'].values

    # ── write ─────────────────────────────────────────────────────────────────

    def log(self, config: dict, metrics: dict) -> str:
        """Append a row and immediately persist to CSV. Returns run_id."""
        run_id = self.make_run_id(config)
        if self.is_done(config):
            print(f'[ExperimentLogger] Already logged run_id={run_id}, skipping.')
            return run_id

        row = {'run_id': run_id, 'timestamp': datetime.now().isoformat()}
        row.update(config)
        row.update(metrics)

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.log_path, index=False)
        print(f'[ExperimentLogger] Logged run_id={run_id} → {self.log_path}')
        return run_id

    # ── read ──────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def pivot_summary(self,
                      index: str = 'ablation_variant',
                      columns: str = 'epsilon',
                      values: str = 'delta_f1_micro') -> pd.DataFrame:
        """Convenience pivot for quick comparison."""
        return self.df.pivot_table(index=index, columns=columns,
                                   values=values, aggfunc='mean')


# ─────────────────────────────────────────────────────────────────────────────
# 8-state correction quality summary
# ─────────────────────────────────────────────────────────────────────────────

def compute_8state_summary(y_true: np.ndarray,
                            y_noisy: np.ndarray,
                            y_corrected: np.ndarray) -> dict:
    """
    Sum 8-state transition counts across all labels.

    State encoding: (true << 2) | (noisy << 1) | corrected_binary
        0 = Keep_TN,  1 = Broke_TN, 2 = Fix_FP,  3 = Miss_FP
        4 = Miss_FN,  5 = Fix_FN,   6 = Broke_TP, 7 = Keep_TP
    """
    T = np.round(np.asarray(y_true,      dtype=float)).astype(int)
    N = np.round(np.asarray(y_noisy,     dtype=float)).astype(int)
    C = np.round(np.asarray(y_corrected, dtype=float)).astype(int)

    codes = (T << 2) | (N << 1) | C          # [N_samples, C_labels]
    flat  = codes.ravel()

    return {
        'keep_tn_total':  int((flat == 0).sum()),
        'broke_tn_total': int((flat == 1).sum()),
        'fix_fp_total':   int((flat == 2).sum()),
        'miss_fp_total':  int((flat == 3).sum()),
        'miss_fn_total':  int((flat == 4).sum()),
        'fix_fn_total':   int((flat == 5).sum()),
        'broke_tp_total': int((flat == 6).sum()),
        'keep_tp_total':  int((flat == 7).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Downstream evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _eval_model(model, data_loader, device) -> dict:
    """Evaluate multi-label model; returns f1/precision/recall/accuracy."""
    from sklearn.metrics import (f1_score, accuracy_score,
                                 precision_score, recall_score)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            b_input_ids  = batch['input_ids'].to(device)
            b_attn_mask  = batch['attention_mask'].to(device)
            b_labels     = batch['labels'].cpu().numpy()
            logits, _    = model(input_ids=b_input_ids,
                                  attention_mask=b_attn_mask)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            all_preds.append(preds)
            all_labels.append(b_labels)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    return {
        'f1_micro':  float(f1_score(y_true, y_pred, average='micro',  zero_division=0)),
        'f1_macro':  float(f1_score(y_true, y_pred, average='macro',  zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        'accuracy':  float(accuracy_score(y_true, y_pred)),
    }


def run_downstream_eval(
    corrected_labels_np: np.ndarray,
    encoded_batch_train: dict,
    data_loader_test: DataLoader,
    model_class,
    args,
    num_labels: int,
    retrain_epochs: int = 10,
    warmup_path: str | None = None,
    init_mode: str = 'warmup',
    loss_mask_np: np.ndarray | None = None,
    save_model_path: str | None = None,
    round_labels: bool = True,
) -> dict:
    """
    Retrain on corrected labels, eval on test set.

    Parameters
    ----------
    corrected_labels_np   : [N_train, C] float ndarray (may contain soft labels 0–1)
    encoded_batch_train   : dict with 'input_ids', 'attention_mask'
    data_loader_test      : DataLoader for test set (uses clean y_true labels)
    model_class           : Mltc | MltcLWAN | MltcLWAN_PerLabel
    args                  : Args instance (needs .device, .batch_size, .encoder_name,
                            .dataset_name, .Noise_ratio, .Noise_type, .epochs)
    num_labels            : number of label classes
    retrain_epochs        : epochs to train on corrected data
    warmup_path           : explicit path to warmup checkpoint (.bin);
                            if None, derived from args
    init_mode             : 'warmup' = load warmup checkpoint; 'scratch' = skip load
    loss_mask_np          : optional [N_train, C] 0/1 mask; 1=參與 BCE, 0=丟棄

    Returns
    -------
    dict with keys: corrected_f1_micro, corrected_f1_macro,
                    precision, recall, accuracy
    """
    from util.dataset import DatasetReassembler

    assert init_mode in ('warmup', 'scratch'), f'init_mode={init_mode}'

    # ── build warmup path ────────────────────────────────────────────────────
    if warmup_path is None:
        noise_suffix = {
            'FP': 'fp', 'FN': 'fn', 'ALL': '',
        }.get(args.Noise_type, '')
        ds   = args.dataset_name.lower()
        enc  = args.encoder_name
        nr   = args.Noise_ratio
        epo  = args.epochs
        if noise_suffix:
            warmup_path = f'./model/warm_model_{ds}_{enc}_noise_{nr}_{noise_suffix}_epoch_{epo}.bin'
        else:
            warmup_path = f'./model/warm_model_{ds}_{enc}_noise_{nr}_epoch_{epo}.bin'

    # ── build retrain DataLoader ─────────────────────────────────────────────
    # round_labels=True  → hard 0/1 (legacy behaviour, used by REL+GMM flip-to-0)
    # round_labels=False → keep soft labels in (0,1) so BCE can use GMM prob as target
    labels_for_retrain = (np.round(corrected_labels_np) if round_labels
                          else corrected_labels_np).astype(np.float32)
    corrected_tensor = torch.tensor(labels_for_retrain, dtype=torch.float32)
    loss_mask_tensor = None
    if loss_mask_np is not None:
        loss_mask_tensor = torch.tensor(loss_mask_np, dtype=torch.float32)

    retrain_loader = DatasetReassembler.create_retraining_loader(
        encoded_inputs=encoded_batch_train,
        corrected_labels=corrected_tensor,
        batch_size=args.batch_size,
        loss_mask=loss_mask_tensor,
    )

    # ── init model ───────────────────────────────────────────────────────────
    model = model_class(num_labels)
    if init_mode == 'warmup':
        if os.path.exists(warmup_path):
            model.load_state_dict(
                torch.load(warmup_path, map_location=args.device)
            )
            print(f'[downstream] Loaded warmup from {warmup_path}')
        else:
            print(f'[downstream] WARNING: warmup checkpoint not found at {warmup_path}, starting from scratch.')
    else:
        print('[downstream] init_mode=scratch — skipping warmup load (BERT pretrained weights kept).')
    model.to(args.device)

    # ── retrain ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    use_mask  = loss_mask_tensor is not None
    criterion = nn.BCEWithLogitsLoss(reduction='none' if use_mask else 'mean')

    model.train()
    for epoch in range(retrain_epochs):
        total_loss = 0.0
        for batch in tqdm(retrain_loader, desc=f'Retrain {epoch+1}/{retrain_epochs}', leave=False):
            b_ids    = batch['input_ids'].to(args.device)
            b_mask   = batch['attention_mask'].to(args.device)
            b_labels = batch['labels'].to(args.device).float()

            model.zero_grad()
            logits, _ = model(input_ids=b_ids, attention_mask=b_mask)

            if use_mask:
                lm = batch['loss_mask'].to(args.device).float()
                loss_per_cell = criterion(logits, b_labels)
                denom = lm.sum().clamp_min(1.0)
                loss  = (loss_per_cell * lm).sum() / denom
            else:
                loss = criterion(logits, b_labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(len(retrain_loader), 1)
        print(f'  Retrain epoch {epoch+1}/{retrain_epochs}  loss={avg:.4f}')
        _wandb_log({'epoch': epoch + 1, 'train_loss': avg})

    # ── evaluate ─────────────────────────────────────────────────────────────
    metrics = _eval_model(model, data_loader_test, args.device)

    # ── save model checkpoint (optional) ─────────────────────────────────────
    if save_model_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_model_path)), exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        print(f'[downstream] Saved retrained model → {save_model_path}')

    # ── cleanup ──────────────────────────────────────────────────────────────
    del model, optimizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'corrected_f1_micro':  metrics['f1_micro'],
        'corrected_f1_macro':  metrics['f1_macro'],
        'precision':           metrics['precision'],
        'recall':              metrics['recall'],
        'accuracy':            metrics['accuracy'],
        'model_path':          save_model_path or '',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Noisy direct baseline (no warmup, no correction — train from BERT pretrained)
# ─────────────────────────────────────────────────────────────────────────────

def eval_noisy_direct_baseline(
    model_class, args, num_labels: int,
    data_loader_train_noisy: DataLoader,
    data_loader_test: DataLoader,
    epochs: int = 10,
    save_model_path: str = None,
) -> dict:
    """
    No-strategy baseline: train from BERT pretrained weights directly on noisy
    labels for `epochs` epochs, then eval on test set.
    """
    model = model_class(num_labels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(data_loader_train_noisy,
                          desc=f'NoisyDirect {epoch+1}/{epochs}', leave=False):
            b_ids    = batch['input_ids'].to(args.device)
            b_mask   = batch['attention_mask'].to(args.device)
            b_labels = batch['labels'].to(args.device).float()

            model.zero_grad()
            logits, _ = model(input_ids=b_ids, attention_mask=b_mask)
            loss = criterion(logits, b_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(len(data_loader_train_noisy), 1)
        print(f'  NoisyDirect epoch {epoch+1}/{epochs}  loss={avg:.4f}')
        _wandb_log({'epoch': epoch + 1, 'train_loss': avg})

    metrics = _eval_model(model, data_loader_test, args.device)

    if save_model_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_model_path)), exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        print(f'[NoisyDirect] Saved model → {save_model_path}')

    del model, optimizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'corrected_f1_micro':  metrics['f1_micro'],
        'corrected_f1_macro':  metrics['f1_macro'],
        'precision':           metrics['precision'],
        'recall':              metrics['recall'],
        'accuracy':            metrics['accuracy'],
        'model_path':          save_model_path or '',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Noisy baseline (warmup model evaluated directly on test set)
# ─────────────────────────────────────────────────────────────────────────────

def eval_noisy_baseline(model_class, args, num_labels: int,
                         data_loader_test: DataLoader,
                         warmup_path: str = None) -> dict:
    """
    Evaluate the warmup model (trained on noisy labels) on the test set.
    Returns dict with noisy_f1_micro, noisy_f1_macro.
    """
    if warmup_path is None:
        noise_suffix = {'FP': 'fp', 'FN': 'fn', 'ALL': ''}.get(args.Noise_type, '')
        ds  = args.dataset_name.lower()
        enc = args.encoder_name
        nr  = args.Noise_ratio
        epo = args.epochs
        if noise_suffix:
            warmup_path = f'./model/warm_model_{ds}_{enc}_noise_{nr}_{noise_suffix}_epoch_{epo}.bin'
        else:
            warmup_path = f'./model/warm_model_{ds}_{enc}_noise_{nr}_epoch_{epo}.bin'

    model = model_class(num_labels)
    if os.path.exists(warmup_path):
        model.load_state_dict(torch.load(warmup_path, map_location=args.device))
    model.to(args.device)

    metrics = _eval_model(model, data_loader_test, args.device)
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'noisy_f1_micro': metrics['f1_micro'],
        'noisy_f1_macro': metrics['f1_macro'],
    }
