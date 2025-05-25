# -*- coding: utf-8 -*-
# @File    : RNADataset.py
# @Author  : Mr.Favian096
# @Description : This file is used to provide model train data
import random
import sys

import torch
import pandas as pd
from ModelConfig import ModelConfig
from torch.utils.data import Dataset

from ModelUtils import *


class PreTrainDataset(Dataset):
    def __init__(
            self,
            train_dataset: list[dict[str: str | int]],
            vocab_path: str = ModelConfig.vocab_file,
            nsp_reverse: bool = ModelConfig.nsp_reverse
    ) -> None:
        """
        read and build dataset for pre-train model
        :param train_dataset: a list of dict, include 'pre', 'post', and 'label'
        :param vocab_path: vocab.txt path
        :param nsp_reverse: whether to reverse the sequence order for nsp
        """
        self.dataset = train_dataset
        self.vocabs: dict[str, int] = {}
        with open(vocab_path, 'r') as file:
            for index, line in enumerate(file):
                self.vocabs[line.replace('\n', '')] = index

        self.nsp_reverse = nsp_reverse

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
            self,
            index: int,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        if isinstance(index, slice):
            index_list: list[int] = [i for i in range(index.start, index.stop, index.step)]
            return [self.token_transform(i) for i in index_list]
        else:
            return self.token_transform(index)

    def token_transform(
            self,
            index: int
    ) -> dict[str, torch.Tensor]:
        pre_seq, post_seq, is_next = self.dataset[index].values()
        pre_token_ids, pre_mlm_labels = self.mask_token(pre_seq)
        post_token_ids, post_mlm_labels = self.mask_token(post_seq)

        token_ids, mlm_labels, segment_labels = self.token_merge(
            pre_token_ids=pre_token_ids,
            pre_mlm_labels=pre_mlm_labels,
            post_token_ids=post_token_ids,
            post_mlm_labels=post_mlm_labels,
        )

        inputs: dict[str, torch.Tensor] = {
            'token_ids': torch.tensor(token_ids),
            'mlm_labels': torch.tensor(mlm_labels),
            'segment_ids': torch.tensor(segment_labels),
            'is_next': torch.tensor(is_next)
        }

        return inputs

    def mask_token(
            self,
            sequence: str
    ) -> tuple[list[int], list[int]]:
        tokens: list[str | int] = get_kmers(sequence)
        mlm_labels: list[int] = []

        for index, token in enumerate(tokens):
            prob = random.random()
            if prob <= 0.15:
                prob /= 0.15
                mlm_label: int = ModelConfig.mask_index

                if prob <= 0.8:
                    tokens[index]: int = mlm_label
                elif prob <= 0.9:
                    mlm_label: int = list(
                        self.vocabs.values())[random.randint(
                        len(ModelConfig.special_tokens),
                        len(self.vocabs) - 1)]
                    tokens[index]: int = mlm_label
                else:
                    mlm_label: int = self.vocabs.get(token, ModelConfig.unk_index)
                    tokens[index]: int = mlm_label
                mlm_labels.append(mlm_label)
            else:
                tokens[index]: int = self.vocabs.get(token, ModelConfig.unk_index)
                mlm_labels.append(0)

        return tokens, mlm_labels

    def token_merge(
            self,
            pre_token_ids: list[int],
            pre_mlm_labels: list[int],
            post_token_ids: list[int],
            post_mlm_labels: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        if self.nsp_reverse:
            pre_token_ids, post_token_ids = post_token_ids, pre_token_ids
            pre_mlm_labels, post_mlm_labels = post_mlm_labels, pre_mlm_labels

        pre_token_ids: list[int] = [ModelConfig.cls_index] + pre_token_ids + [ModelConfig.sep_index]
        post_token_ids: list[int] = post_token_ids + [ModelConfig.sep_index]

        pre_mlm_labels: list[int] = [ModelConfig.pad_index] + pre_mlm_labels + [ModelConfig.pad_index]
        post_mlm_labels: list[int] = post_mlm_labels + [ModelConfig.pad_index]

        segment_labels: list[int] = ([1 for _ in range(len(pre_token_ids))] +
                                     [2 for _ in range(len(post_token_ids))])[:ModelConfig.token_num]
        token_ids: list[int] = (pre_token_ids + post_token_ids)[:ModelConfig.token_num]
        mlm_labels: list[int] = (pre_mlm_labels + post_mlm_labels)[:ModelConfig.token_num]

        padding: list[int] = [ModelConfig.pad_index for _ in range(ModelConfig.token_num - len(token_ids))]
        token_ids.extend(padding)
        mlm_labels.extend(padding)
        segment_labels.extend(padding)
        return token_ids, mlm_labels, segment_labels


class TuneDataset(PreTrainDataset):
    def __init__(
            self,
            tune_dataset: list[dict[str: str | int]],
            vocab_path: str = ModelConfig.vocab_file,
            nsp_reverse: bool = ModelConfig.nsp_reverse
    ) -> None:
        """
        read and build dataset for fine tune pre-train model
        :param tune_dataset: a list of dict, include 'pre', 'post', and 'label'
        :param vocab_path: vocab.txt path
        :param nsp_reverse: whether to reverse the sequence order for nsp
        """
        super().__init__(
            train_dataset=tune_dataset,
            vocab_path=vocab_path,
            nsp_reverse=nsp_reverse)

    def token_transform(
            self,
            index: int
    ) -> dict[str, torch.Tensor]:
        pre_seq, post_seq, label = self.dataset[index].values()
        pre_token_ids = self.mask_token(pre_seq)
        post_token_ids = self.mask_token(post_seq)

        token_ids, segment_labels = self.tune_token_merge(
            pre_token_ids=pre_token_ids,
            post_token_ids=post_token_ids)

        inputs = {
            'token_ids': torch.tensor(token_ids),
            'segment_ids': torch.tensor(segment_labels),
            'label': torch.tensor(label)
        }

        return inputs

    def mask_token(
            self,
            sequence: str
    ) -> list[int]:
        tokens: list[str | int] = get_kmers(sequence)
        for index, token in enumerate(tokens):
            tokens[index] = self.vocabs.get(token, ModelConfig.unk_index)
        return tokens

    def tune_token_merge(
            self,
            pre_token_ids: list[int],
            post_token_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        if self.nsp_reverse:
            pre_token_ids, post_token_ids = post_token_ids, pre_token_ids

        pre_token_ids = [ModelConfig.cls_index] + pre_token_ids + [ModelConfig.sep_index]
        post_token_ids = post_token_ids + [ModelConfig.sep_index]

        segment_labels = ([1 for _ in range(len(pre_token_ids))] +
                          [2 for _ in range(len(post_token_ids))])[:ModelConfig.token_num]
        token_ids = (pre_token_ids + post_token_ids)[:ModelConfig.token_num]

        padding = [ModelConfig.pad_index for _ in range(ModelConfig.token_num - len(token_ids))]
        token_ids.extend(padding)
        segment_labels.extend(padding)

        return token_ids, segment_labels
