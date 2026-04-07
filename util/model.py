import torch.nn as nn
import torch

from transformers import (
    BertTokenizer, BertConfig, BertModel,
    RobertaModel, RobertaConfig, RobertaTokenizer,
    XLNetTokenizer, XLNetModel, XLNetConfig,
    RobertaTokenizerFast
)
from torch.autograd import Variable

import torch.nn.functional as F
# model 

class Mltc(nn.Module):
    """
    Baseline Model: 使用 [CLS] token 的多標籤文本分類模型

    這是您原始的模型，使用 [CLS] token 作為文件表示。
    所有 label 共享相同的 representation space。
    """

    def __init__(self, num_labels):
        super(Mltc, self).__init__()
        # model_config = BertConfig.from_pretrained("bert-base-uncased")
        # bert = BertModel.from_pretrained("bert-base-uncased", config=model_config)
        self.encoder =BertModel.from_pretrained("bert-base-uncased")

        self.feature_layers = 1


        concatenated_dim = self.encoder.config.hidden_size * self.feature_layers
        # self.feature_layer = nn.Linear(concatenated_dim, 128)
        # 分類層的輸入維度是 feature_layer 的輸出維度
        self.classifier = nn.Linear(concatenated_dim, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None,labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token 的輸出即為整個句子的特徵
        # outputs[0] 的 shape 是 (batch_size, sequence_length, hidden_size)
        # 我們取第一個 token ([CLS]) 的 hidden state
        features = outputs[0][:, 0, :]
        # features = self.dropout(features)
        logits = self.classifier(features)


        return logits,features


class MltcLWAN(nn.Module):
    """
    Proposed Model: Label-Wise Attention Network (LWAN)

    核心改進：每個 label 有自己的文件表示 d_l，而非共享 [CLS]

    三階段架構：
        Stage A: v_t = tanh(W·h_t + b)  【共用投影】
        Stage B: a_lt = softmax(v_t^T · u_l)  【per-label attention】
        Stage C: d_l = Σ_t a_lt · h_t  【per-label representation】

    與 Mltc 的差異：
        - Mltc: 使用 [CLS] token，所有 label 共享
        - MltcLWAN: 使用 label-wise attention，每個 label 獨立
    """

    def __init__(self, num_labels, bert_name="bert-base-uncased", attention_dim=None):
        super(MltcLWAN, self).__init__()
        self.num_labels = num_labels

        # BERT encoder
        self.encoder = BertModel.from_pretrained(bert_name)
        hidden_size = self.encoder.config.hidden_size

        # Attention dimension (預設與 hidden_size 相同)
        self.attention_dim = attention_dim if attention_dim is not None else hidden_size

        # Stage A: 共用投影層 W, b
        self.W = nn.Linear(hidden_size, self.attention_dim, bias=True)
        nn.init.xavier_uniform_(self.W.weight)

        # Stage B: Per-label query vectors U
        # U: [num_labels, attention_dim]
        self.U = nn.Parameter(torch.randn(num_labels, self.attention_dim))
        nn.init.xavier_normal_(self.U)

        # Classifier: 將 d_l 映射到 logit
        self.classifier = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: optional
            labels: optional

        Returns:
            logits: [batch_size, num_labels]
            D: [batch_size, num_labels, hidden_size] - per-label representations
        """
        # BERT encoding - 保留所有 token
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        h = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Stage A: 共用投影
        v = torch.tanh(self.W(h))  # [batch_size, seq_len, attention_dim]

        # Stage B: Per-label attention scores
        # einsum: 'bth,lh->btl'
        # b=batch, t=token, h=attention_dim, l=label
        scores = torch.einsum('bth,lh->btl', v, self.U)  # [batch_size, seq_len, num_labels]

        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax over tokens for each label
        alpha = torch.softmax(scores, dim=1)  # [batch_size, seq_len, num_labels]

        # Stage C: Label-specific representations
        # einsum: 'btl,bth->blh'
        # d_l = Σ_t alpha_lt · h_t
        D = torch.einsum('btl,bth->blh', alpha, h)  # [batch_size, num_labels, hidden_size]

        # Classification
        logits = self.classifier(D).squeeze(-1)  # [batch_size, num_labels]

        return logits, D


class MltcLWAN_PerLabel(nn.Module):
    """
    Advanced Model: Per-Label Independent Attention Network

    與 MltcLWAN 的核心差異:
        MltcLWAN:         所有 label 共享同一個 W (shared projection)
        MltcLWAN_PerLabel: 每個 label 有獨立的 W_l (per-label projection)

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
        ✓ 最適合 per-label noise detection

    參數成本:
        × 參數量增加約 28% (相較於 MltcLWAN)
        × 約 31M 額外參數 (對於 54 labels)
    """

    def __init__(self, num_labels, bert_name="bert-base-uncased", attention_dim=None):
        super(MltcLWAN_PerLabel, self).__init__()
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
            token_type_ids: optional
            labels: optional

        Returns:
            logits: [batch_size, num_labels]
            D: [batch_size, num_labels, hidden_size] - per-label representations
        """
        batch_size = input_ids.size(0)

        # BERT Encoding
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        h = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # ============================================
        # Stage A + B + C: Per-Label 獨立計算
        # ============================================
        D_list = []

        for label_idx in range(self.num_labels):
            # Stage A: 使用該 label 的專屬投影 W_l
            v_l = torch.tanh(self.W_list[label_idx](h))  # [batch_size, seq_len, attention_dim]

            # Stage B: 計算該 label 的 attention scores
            u_l = self.U[label_idx]  # [attention_dim]
            scores_l = torch.matmul(v_l, u_l)  # [batch_size, seq_len]

            # Mask padding tokens
            scores_l = scores_l.masked_fill(attention_mask == 0, -1e9)

            # Softmax over tokens
            alpha_l = torch.softmax(scores_l, dim=1)  # [batch_size, seq_len]

            # Stage C: Label-specific representation
            # d_l = Σ_t alpha_lt · h_t
            d_l = torch.matmul(alpha_l.unsqueeze(1), h).squeeze(1)  # [batch_size, hidden_size]

            D_list.append(d_l)

        # Stack all label representations
        D = torch.stack(D_list, dim=1)  # [batch_size, num_labels, hidden_size]

        # Classification
        logits = self.classifier(D).squeeze(-1)  # [batch_size, num_labels]

        return logits, D
