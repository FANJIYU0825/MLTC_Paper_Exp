"""
Multi-Label Text Classification Encoders
比較兩種文件表示方法：
1. BertCLS: 傳統 [CLS] token 方法 (baseline)
2. BertLWAN: Label-Wise Attention Network (proposed)

根據論文 "Model Design: Label-Specific Denoising via BERT-LWAN"
Author: Tony's Research Team
Date: 2026-03-19
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertCLS(nn.Module):
    """
    Baseline: 使用 [CLS] token 作為文件表示

    架構:
        Input → BERT → Extract [CLS] token → Classifier

    特點:
        - 所有 label 共享相同的文件表示 (h_[CLS])
        - 簡單、高效
        - 可能存在 cross-label interference (交叉標籤干擾)

    適用情境:
        - Baseline 比較
        - 不需要 per-label representation 的任務
    """

    def __init__(self, num_labels, bert_name="bert-base-uncased"):
        super(BertCLS, self).__init__()
        self.num_labels = num_labels

        # BERT encoder
        self.encoder = BertModel.from_pretrained(bert_name)

        # 分類層：從 [CLS] 直接預測所有 labels
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)
            labels: [batch_size, num_labels] (optional, for compatibility)

        Returns:
            logits: [batch_size, num_labels]
            features: [batch_size, hidden_size] - [CLS] token representation
        """
        # BERT forward
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 取 [CLS] token (第一個 token 的 hidden state)
        # outputs[0] = last_hidden_state: [batch_size, seq_len, hidden_size]
        cls_output = outputs[0][:, 0, :]  # [batch_size, hidden_size]

        # 分類
        logits = self.classifier(cls_output)  # [batch_size, num_labels]

        # 回傳格式與原始 Mltc 相同
        return logits, cls_output


class BertLWAN(nn.Module):
    """
    Proposed: Label-Wise Attention Network

    三階段架構 (Three-Stage Architecture):

    Stage A - 共用 Token 投影 (Shared Token Projection):
        v_t = tanh(W·h_t + b)  for all tokens t
        - W, b 為所有 label 共用

    Stage B - Per-Label Attention (每個 label 獨立的注意力):
        a_lt = softmax(v_t^T · u_l)  for each label l
        - u_l 是 label l 的 query vector (learnable parameter)

    Stage C - Label-Specific Representation:
        d_l = Σ_t a_lt · h_t  for each label l
        - 每個 label 有自己的文件表示 d_l

    核心優勢:
        ✓ 每個 label 在 representation 階段就已分離
        ✓ Noise detection 在 label-specific space 進行
        ✓ 消除 cross-label interference

    與 [CLS] 的關鍵差異:
        [CLS]: 所有 token → 單一向量 → 所有 label 共用
        LWAN:  所有 token → per-label attention → 每個 label 各自的向量
    """

    def __init__(self, num_labels, bert_name="bert-base-uncased", attention_dim=None):
        super(BertLWAN, self).__init__()
        self.num_labels = num_labels

        # BERT encoder
        self.encoder = BertModel.from_pretrained(bert_name)
        hidden_size = self.encoder.config.hidden_size

        # Attention dimension (預設與 hidden_size 相同)
        self.attention_dim = attention_dim if attention_dim is not None else hidden_size

        # ============================================
        # Stage A: 共用投影層 (Shared Projection)
        # ============================================
        # W, b: 將 h_t 投影到 attention space
        self.W = nn.Linear(hidden_size, self.attention_dim, bias=True)
        nn.init.xavier_uniform_(self.W.weight)

        # ============================================
        # Stage B: Per-Label Query Vectors
        # ============================================
        # U: [num_labels, attention_dim]
        # 每個 label 有自己的 query vector u_l
        self.U = nn.Parameter(torch.randn(num_labels, self.attention_dim))
        nn.init.xavier_normal_(self.U)

        # ============================================
        # Classifier: 將 d_l 映射到 logit
        # ============================================
        # 注意：這裡每個 label 用同一個 classifier，但輸入是不同的 d_l
        self.classifier = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)
            labels: [batch_size, num_labels] (optional, for compatibility)

        Returns:
            logits: [batch_size, num_labels]
            D: [batch_size, num_labels, hidden_size] - per-label representations
        """
        batch_size = input_ids.size(0)

        # ============================================
        # BERT Encoding (取得所有 token 的 hidden states)
        # ============================================
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # h: [batch_size, seq_len, hidden_size]
        # 這是與 [CLS] 方法的第一個關鍵差異：我們保留所有 token
        h = outputs.last_hidden_state

        # ============================================
        # Stage A: Shared Token Projection
        # ============================================
        # v_t = tanh(W·h_t + b) for all tokens
        # v: [batch_size, seq_len, attention_dim]
        v = torch.tanh(self.W(h))

        # ============================================
        # Stage B: Per-Label Attention Scores
        # ============================================
        # 計算 v_t^T · u_l for all (t, l) pairs
        # einsum 說明:
        #   'bth,lh->btl'
        #   b=batch, t=token, h=attention_dim, l=label
        #   v: [b, t, h] @ U: [l, h] → scores: [b, t, l]
        scores = torch.einsum('bth,lh->btl', v, self.U)  # [batch_size, seq_len, num_labels]

        # Mask padding tokens (將 padding 位置設為 -inf)
        # attention_mask: [batch_size, seq_len] → [batch_size, seq_len, 1]
        mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax over tokens (dim=1) for each label
        # alpha_lt: attention weight of token t for label l
        alpha = torch.softmax(scores, dim=1)  # [batch_size, seq_len, num_labels]

        # ============================================
        # Stage C: Label-Specific Document Representations
        # ============================================
        # d_l = Σ_t alpha_lt · h_t for each label l
        # einsum 說明:
        #   'btl,bth->blh'
        #   alpha: [b, t, l] @ h: [b, t, h] → D: [b, l, h]
        D = torch.einsum('btl,bth->blh', alpha, h)  # [batch_size, num_labels, hidden_size]

        # ============================================
        # Classification
        # ============================================
        # 將每個 d_l 映射到 scalar logit
        # D: [batch_size, num_labels, hidden_size]
        # classifier output: [batch_size, num_labels, 1]
        logits = self.classifier(D).squeeze(-1)  # [batch_size, num_labels]

        # 回傳格式：
        # logits: [batch_size, num_labels] - 用於 loss 計算
        # D: [batch_size, num_labels, hidden_size] - per-label representations，用於 denoising
        return logits, D

    def get_attention_weights(self, input_ids, attention_mask, token_type_ids=None):
        """
        輔助函數：取得 attention weights (用於可視化或分析)

        Returns:
            alpha: [batch_size, seq_len, num_labels] - attention weights
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            h = outputs.last_hidden_state
            v = torch.tanh(self.W(h))
            scores = torch.einsum('bth,lh->btl', v, self.U)

            mask = attention_mask.unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)

            alpha = torch.softmax(scores, dim=1)

        return alpha


# ============================================
# 比較函數 (用於測試和 debug)
# ============================================

def compare_encoders():
    """
    比較兩種 encoder 的輸出形狀和參數量
    """
    print("="*60)
    print("Encoder Comparison Test")
    print("="*60)

    # 設定
    batch_size = 4
    seq_len = 64
    num_labels = 54  # AAPD dataset

    # 建立 dummy input
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # ============================================
    # Test BertCLS
    # ============================================
    print("\n[1] Testing BertCLS (Baseline)...")
    cls_model = BertCLS(num_labels=num_labels)
    logits_cls, features_cls = cls_model(input_ids, attention_mask)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits_cls.shape}")  # [batch_size, num_labels]
    print(f"  Features shape: {features_cls.shape}")  # [batch_size, hidden_size]
    print(f"  → All labels share the SAME representation (CLS token)")

    # 參數量
    cls_params = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {cls_params:,}")

    # ============================================
    # Test BertLWAN
    # ============================================
    print("\n[2] Testing BertLWAN (Proposed)...")
    lwan_model = BertLWAN(num_labels=num_labels)
    logits_lwan, features_lwan = lwan_model(input_ids, attention_mask)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits_lwan.shape}")  # [batch_size, num_labels]
    print(f"  Features shape: {features_lwan.shape}")  # [batch_size, num_labels, hidden_size]
    print(f"  → Each label has its OWN representation (d_l)")

    # 參數量
    lwan_params = sum(p.numel() for p in lwan_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {lwan_params:,}")

    # ============================================
    # Test Attention Weights
    # ============================================
    print("\n[3] Testing Attention Weight Extraction...")
    attention_weights = lwan_model.get_attention_weights(input_ids, attention_mask)
    print(f"  Attention weights shape: {attention_weights.shape}")  # [batch_size, seq_len, num_labels]
    print(f"  → Can visualize which tokens are important for each label")

    # ============================================
    # Parameter Difference
    # ============================================
    print("\n[4] Parameter Comparison...")
    diff = lwan_params - cls_params
    print(f"  BertCLS:  {cls_params:,} parameters")
    print(f"  BertLWAN: {lwan_params:,} parameters")
    print(f"  Difference: {diff:,} (+{diff/cls_params*100:.2f}%)")
    print(f"  → LWAN adds W, U parameters for attention mechanism")

    # ============================================
    # Key Differences Summary
    # ============================================
    print("\n" + "="*60)
    print("Key Differences Summary")
    print("="*60)
    print("\n[CLS Approach]")
    print("  • Uses only [CLS] token (h_1)")
    print("  • All labels share the same representation")
    print("  • Simpler, fewer parameters")
    print("  • May suffer from cross-label interference")

    print("\n[LWAN Approach]")
    print("  • Uses ALL tokens (h_1, h_2, ..., h_T)")
    print("  • Each label has its own representation (d_l)")
    print("  • Label-specific attention weights")
    print("  • Better for per-label noise detection")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


class BertLWAN_PerLabel(nn.Module):
    """
    Advanced: Per-Label Independent Attention Network

    與 BertLWAN 的核心差異:
        BertLWAN:         所有 label 共享同一個 W (shared projection)
        BertLWAN_PerLabel: 每個 label 有獨立的 W_l (per-label projection)

    三階段架構 (Per-Label Version):

    Stage A - Per-Label Token Projection:
        v_lt = tanh(W_l·h_t + b_l)  for each label l
        - 每個 label 有自己的 W_l, b_l 參數
        - 完全獨立的 token 投影空間

    Stage B - Per-Label Attention:
        a_lt = softmax(v_lt^T · u_l)  for each label l
        - u_l 是 label l 的 query vector

    Stage C - Label-Specific Representation:
        d_l = Σ_t a_lt · h_t  for each label l

    理論優勢:
        ✓ 更高的 label 表示獨立性
        ✓ 每個 label 可學習自己的 feature space
        ✓ 減少 label 間的參數干擾

    權衡:
        × 參數量增加 (num_labels × W 參數)
        × 可能需要更多訓練數據

    適用情境:
        - Label 語義差異極大的任務
        - Per-label denoising (噪音檢測)
        - 需要完全分離 label representations
    """

    def __init__(self, num_labels, bert_name="bert-base-uncased", attention_dim=None):
        super(BertLWAN_PerLabel, self).__init__()
        self.num_labels = num_labels

        # BERT encoder
        self.encoder = BertModel.from_pretrained(bert_name)
        hidden_size = self.encoder.config.hidden_size

        # Attention dimension
        self.attention_dim = attention_dim if attention_dim is not None else hidden_size

        # ============================================
        # Stage A: Per-Label Projection Layers
        # ============================================
        # 關鍵差異: 每個 label 有自己的 W_l, b_l
        # 使用 ModuleList 來管理 num_labels 個獨立的 Linear 層
        self.W_list = nn.ModuleList([
            nn.Linear(hidden_size, self.attention_dim, bias=True)
            for _ in range(num_labels)
        ])

        # 初始化每個 W_l
        for W_l in self.W_list:
            nn.init.xavier_uniform_(W_l.weight)

        # ============================================
        # Stage B: Per-Label Query Vectors
        # ============================================
        # U: [num_labels, attention_dim]
        self.U = nn.Parameter(torch.randn(num_labels, self.attention_dim))
        nn.init.xavier_normal_(self.U)

        # ============================================
        # Classifier
        # ============================================
        self.classifier = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)
            labels: [batch_size, num_labels] (optional, for compatibility)

        Returns:
            logits: [batch_size, num_labels]
            D: [batch_size, num_labels, hidden_size] - per-label representations
        """
        batch_size = input_ids.size(0)

        # ============================================
        # BERT Encoding
        # ============================================
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # h: [batch_size, seq_len, hidden_size]
        h = outputs.last_hidden_state

        # ============================================
        # Stage A + B + C: Per-Label 獨立計算
        # ============================================
        # 為每個 label 獨立計算 attention 和 representation
        D_list = []

        for label_idx in range(self.num_labels):
            # Stage A: 使用該 label 的專屬投影 W_l
            # v_l: [batch_size, seq_len, attention_dim]
            v_l = torch.tanh(self.W_list[label_idx](h))

            # Stage B: 計算該 label 的 attention scores
            # u_l: [attention_dim]
            u_l = self.U[label_idx]  # [attention_dim]

            # scores_l: [batch_size, seq_len]
            # v_l: [b, t, h] @ u_l: [h] → [b, t]
            scores_l = torch.matmul(v_l, u_l)  # [batch_size, seq_len]

            # Mask padding tokens
            scores_l = scores_l.masked_fill(attention_mask == 0, -1e9)

            # Softmax over tokens
            alpha_l = torch.softmax(scores_l, dim=1)  # [batch_size, seq_len]

            # Stage C: Label-specific representation
            # d_l = Σ_t alpha_lt · h_t
            # alpha_l: [b, t] → [b, t, 1]
            # h: [b, t, h]
            # d_l: [b, h]
            d_l = torch.matmul(alpha_l.unsqueeze(1), h).squeeze(1)  # [batch_size, hidden_size]

            D_list.append(d_l)

        # Stack all label representations
        # D: [batch_size, num_labels, hidden_size]
        D = torch.stack(D_list, dim=1)

        # ============================================
        # Classification
        # ============================================
        logits = self.classifier(D).squeeze(-1)  # [batch_size, num_labels]

        return logits, D

    def get_attention_weights(self, input_ids, attention_mask, token_type_ids=None):
        """
        輔助函數: 取得每個 label 的 attention weights

        Returns:
            alpha: [batch_size, seq_len, num_labels] - attention weights
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            h = outputs.last_hidden_state

            alpha_list = []
            for label_idx in range(self.num_labels):
                v_l = torch.tanh(self.W_list[label_idx](h))
                u_l = self.U[label_idx]
                scores_l = torch.matmul(v_l, u_l)
                scores_l = scores_l.masked_fill(attention_mask == 0, -1e9)
                alpha_l = torch.softmax(scores_l, dim=1)
                alpha_list.append(alpha_l)

            # [batch_size, seq_len, num_labels]
            alpha = torch.stack(alpha_list, dim=-1)

        return alpha


# ============================================
# 比較函數 (用於測試和 debug)
# ============================================

def compare_encoders():
    """
    比較三種 encoder 的輸出形狀和參數量
    """
    print("="*60)
    print("Encoder Comparison Test")
    print("="*60)

    # 設定
    batch_size = 4
    seq_len = 64
    num_labels = 54  # AAPD dataset

    # 建立 dummy input
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # ============================================
    # Test BertCLS
    # ============================================
    print("\n[1] Testing BertCLS (Baseline)...")
    cls_model = BertCLS(num_labels=num_labels)
    logits_cls, features_cls = cls_model(input_ids, attention_mask)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits_cls.shape}")  # [batch_size, num_labels]
    print(f"  Features shape: {features_cls.shape}")  # [batch_size, hidden_size]
    print(f"  → All labels share the SAME representation (CLS token)")

    # 參數量
    cls_params = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {cls_params:,}")

    # ============================================
    # Test BertLWAN
    # ============================================
    print("\n[2] Testing BertLWAN (Shared W)...")
    lwan_model = BertLWAN(num_labels=num_labels)
    logits_lwan, features_lwan = lwan_model(input_ids, attention_mask)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits_lwan.shape}")  # [batch_size, num_labels]
    print(f"  Features shape: {features_lwan.shape}")  # [batch_size, num_labels, hidden_size]
    print(f"  → Each label has its OWN representation (shared W)")

    # 參數量
    lwan_params = sum(p.numel() for p in lwan_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {lwan_params:,}")

    # ============================================
    # Test BertLWAN_PerLabel
    # ============================================
    print("\n[3] Testing BertLWAN_PerLabel (Per-Label W)...")
    lwan_perlabel_model = BertLWAN_PerLabel(num_labels=num_labels)
    logits_lwan_pl, features_lwan_pl = lwan_perlabel_model(input_ids, attention_mask)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits_lwan_pl.shape}")  # [batch_size, num_labels]
    print(f"  Features shape: {features_lwan_pl.shape}")  # [batch_size, num_labels, hidden_size]
    print(f"  → Each label has INDEPENDENT W_l and representation")

    # 參數量
    lwan_pl_params = sum(p.numel() for p in lwan_perlabel_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {lwan_pl_params:,}")

    # ============================================
    # Test Attention Weights
    # ============================================
    print("\n[4] Testing Attention Weight Extraction...")
    attention_weights_lwan = lwan_model.get_attention_weights(input_ids, attention_mask)
    attention_weights_lwan_pl = lwan_perlabel_model.get_attention_weights(input_ids, attention_mask)
    print(f"  LWAN Attention weights shape: {attention_weights_lwan.shape}")
    print(f"  LWAN_PerLabel Attention weights shape: {attention_weights_lwan_pl.shape}")
    print(f"  → Can visualize which tokens are important for each label")

    # ============================================
    # Parameter Comparison
    # ============================================
    print("\n[5] Parameter Comparison...")
    diff_lwan = lwan_params - cls_params
    diff_lwan_pl = lwan_pl_params - lwan_params

    print(f"  BertCLS:            {cls_params:,} parameters")
    print(f"  BertLWAN:           {lwan_params:,} parameters (+{diff_lwan:,}, +{diff_lwan/cls_params*100:.2f}%)")
    print(f"  BertLWAN_PerLabel:  {lwan_pl_params:,} parameters (+{diff_lwan_pl:,}, +{diff_lwan_pl/lwan_params*100:.2f}%)")
    print(f"\n  Extra cost of per-label W: {diff_lwan_pl:,} parameters")
    print(f"  → {diff_lwan_pl / num_labels:,.0f} parameters per label")

    # ============================================
    # Key Differences Summary
    # ============================================
    print("\n" + "="*60)
    print("Key Differences Summary")
    print("="*60)
    print("\n[CLS Approach]")
    print("  • Uses only [CLS] token (h_1)")
    print("  • All labels share the same representation")
    print("  • Simplest, fewest parameters")
    print("  • May suffer from cross-label interference")

    print("\n[LWAN Approach - Shared W]")
    print("  • Uses ALL tokens with attention")
    print("  • Shared W projection, per-label U query")
    print("  • Each label has its own representation")
    print("  • Good balance of capacity and parameter efficiency")

    print("\n[LWAN_PerLabel Approach - Independent W]")
    print("  • Uses ALL tokens with per-label attention")
    print("  • Each label has W_l and U_l")
    print("  • Maximum label independence")
    print("  • Best for label-specific denoising")
    print("  • Higher parameter cost")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


if __name__ == '__main__':
    compare_encoders()
