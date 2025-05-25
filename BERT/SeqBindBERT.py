# -*- coding: utf-8 -*-
# @File    : RNABindBERT.py
# @Author  : Mr.Favian096
# @Description : This file provide base bert model

import torch.nn as nn
from torch.nn.modules.module import T
from ModelConfig import ModelConfig


class SeqBindBERT(nn.Module):
    def __init__(self, bert, hidden=ModelConfig.model_hidden):
        super().__init__()

        self.bert = bert
        self.linear: nn.Linear = nn.Linear(hidden, 2)
        self.softmax: nn.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, token_ids, segment_ids):
        hidden = self.bert(token_ids, segment_ids)[:, 0, :]
        return self.softmax(self.linear(hidden))
