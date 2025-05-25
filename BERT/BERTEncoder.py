# -*- coding: utf-8 -*-
# @File    : BERTEncoder.py
# @Author  : Mr.Favian096
# @Description : This file is used to implement bert model's encoder layer

import math

import torch
import torch.nn as nn
import torch.nn.functional as functions

from ModelConfig import ModelConfig


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = functions.softmax

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask=None,
            dropout=None) -> torch.Tensor:
        scores: torch.Tensor = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores: torch.Tensor = scores.masked_fill(mask == 0, -1e9)

        prob_scores: torch.Tensor = self.softmax(scores, dim=-1)

        if dropout is not None:
            prob_scores: torch.Tensor = dropout(prob_scores)

        return torch.matmul(prob_scores, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads=ModelConfig.attention_head_num, model_dim=ModelConfig.model_hidden, dropout=0.1):
        super().__init__()
        assert model_dim % heads == 0

        self.matrix_dim: int = model_dim // heads
        self.heads: int = heads

        self.linear_layers: nn.ModuleList = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(3)])
        self.output_linear: nn.Linear = nn.Linear(model_dim, model_dim)
        self.attention: SelfAttention = SelfAttention()

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask=None) -> torch.Tensor:
        batch_size: int = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.heads, self.matrix_dim).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        attention: torch.Tensor = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        attention: torch.Tensor = attention.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.heads * self.matrix_dim)

        return self.output_linear(attention)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale: nn.Parameter = nn.Parameter(torch.ones(features))
        self.move: nn.Parameter = nn.Parameter(torch.zeros(features))
        self.eps: float = eps

    def forward(self, hidden):
        mean: torch.Tensor = hidden.mean(-1, keepdim=True)
        std: torch.Tensor = hidden.std(-1, keepdim=True)
        return self.scale * (hidden - mean) / (std + self.eps) + self.move


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm: LayerNorm = LayerNorm(size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
            self,
            hidden: torch.Tensor,
            sublayer
    ) -> torch.Tensor:
        return hidden + self.dropout(sublayer(self.norm(hidden)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.gelu(hidden)


class FeedForward(nn.Module):
    def __init__(
            self,
            model_dim,
            feedforward_dim,
            dropout=ModelConfig.feed_forward_dropout
    ):
        super(FeedForward, self).__init__()
        self.expand = nn.Linear(model_dim, feedforward_dim)
        self.shrink = nn.Linear(feedforward_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.shrink(self.dropout(self.activation(self.expand(hidden))))


class EncoderLayer(nn.Module):
    def __init__(
            self,
            hidden: int,
            attention_heads: int,
            feed_forward_hidden: int,
            dropout: ModelConfig.encoder_dropout):
        super().__init__()
        self.attention: MultiHeadAttention = MultiHeadAttention(heads=attention_heads,
                                                                model_dim=hidden)
        self.feed_forward: FeedForward = FeedForward(model_dim=hidden,
                                                     feedforward_dim=feed_forward_hidden,
                                                     dropout=dropout)
        self.input_sublayer: SublayerConnection = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer: SublayerConnection = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(self, embeddings, mask):
        hidden = self.input_sublayer(embeddings, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)
