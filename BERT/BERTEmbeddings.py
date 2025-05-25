# -*- coding: utf-8 -*-
# @File    : BERTEmbeddings.py
# @Author  : Mr.Favian096
# @Description : This file is used to process lncRNA and miRNA sequence embeddings, include token embeddings, segment embeddings and position embeddings

import math
import torch
import torch.nn as nn

from ModelConfig import ModelConfig


class TokenEmbeddings(nn.Embedding):
    def __init__(self, vocab_size, embed_size=ModelConfig.embedding_size):
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=ModelConfig.pad_index
        )


class SegmentEmbeddings(nn.Embedding):
    def __init__(self, embed_size=ModelConfig.embedding_size):
        super().__init__(
            num_embeddings=3,
            embedding_dim=embed_size,
            padding_idx=ModelConfig.pad_index
        )


class PositionEmbeddings(nn.Module):
    def __init__(self, model_dim, token_num=ModelConfig.token_num):
        super().__init__()
        position_embed: torch.Tensor = torch.zeros(token_num, model_dim).float()
        position_embed.requires_grad = False

        position: torch.Tensor = torch.arange(0, token_num).float().unsqueeze(1)
        div_term: torch.Tensor = (torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)).exp()

        position_embed[:, 0::2] = torch.sin(position * div_term)
        position_embed[:, 1::2] = torch.cos(position * div_term)

        position_embed: torch.Tensor = position_embed.unsqueeze(0)
        self.register_buffer('position_embed', position_embed)

    def forward(self, token_ids):
        return self.position_embed[:, :token_ids.size(1)].expand(token_ids.size(0), -1, -1)


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_size: int = ModelConfig.embedding_size,
            dropout: float = ModelConfig.embedding_dropout
    ):
        super().__init__()
        self.token: TokenEmbeddings = TokenEmbeddings(vocab_size=vocab_size, embed_size=embed_size)
        self.segment: SegmentEmbeddings = SegmentEmbeddings(embed_size=embed_size)
        self.position: PositionEmbeddings = PositionEmbeddings(model_dim=embed_size)

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids: torch.Tensor, segment_label):
        embeddings: torch.Tensor = self.token(token_ids) + self.segment(segment_label) + self.position(token_ids)
        return self.dropout(embeddings)
