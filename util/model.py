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
