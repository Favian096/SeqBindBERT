# -*- coding: utf-8 -*-
# @File    : BERTBase.py
# @Author  : Mr.Favian096
# @Description : This file is used to comform BERT model Embedding layer Encoder layer
import torch
import torch.nn as nn

from ModelConfig import ModelConfig
from BERTEncoder import EncoderLayer
from BERTEmbeddings import EmbeddingLayer


class BERT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden: int = ModelConfig.model_hidden,
            encoder_num: int = ModelConfig.encoder_num,
            attention_heads: int = ModelConfig.attention_head_num,
            dropout: float = ModelConfig.encoder_dropout
    ):
        super().__init__()
        self.hidden: int = hidden
        self.layer_num: int = encoder_num
        self.attention_heads: int = attention_heads
        self.dropout: float = dropout

        self.feed_forward_hidden: int = hidden * 4

        self.embedding: EmbeddingLayer = EmbeddingLayer(
            vocab_size=vocab_size,
            embed_size=hidden)

        self.encoder_layers: nn.ModuleList = nn.ModuleList([
            EncoderLayer(hidden, attention_heads, self.feed_forward_hidden, self.dropout)
            for _ in range(self.layer_num)])

    def forward(
            self,
            token_ids: torch.Tensor,
            segment_ids: torch.Tensor
    ) -> torch.Tensor:
        mask = (token_ids > 0).unsqueeze(1).repeat(1, token_ids.size(1), 1).unsqueeze(1)

        embeddings = self.embedding(token_ids, segment_ids)

        for encoder in self.encoder_layers:
            hidden = encoder.forward(embeddings, mask)

            return hidden


class MaskedLanguageModel(nn.Module):
    def __init__(
            self,
            hidden: int,
            vocab_size: int
    ):
        super().__init__()
        self.linear: nn.Linear = nn.Linear(hidden, vocab_size)
        self.softmax: nn.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden):
        return self.softmax(self.linear(hidden))


class NextSentencePrediction(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear: nn.Linear = nn.Linear(hidden, 2)
        self.softmax: nn.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden):
        return self.softmax(self.linear(hidden[:, 0]))


class BERTPreTrain(nn.Module):
    def __init__(self, bert: BERT, vocab_size: int):
        super().__init__()
        self.bert: BERT = bert
        self.nsp: NextSentencePrediction = NextSentencePrediction(self.bert.hidden)
        self.mlm: MaskedLanguageModel = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, token_ids, segment_ids):
        hidden: torch.Tensor = self.bert(token_ids, segment_ids)
        return self.nsp(hidden), self.mlm(hidden)
