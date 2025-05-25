# -*- coding: utf-8 -*-
# @File    : ModelUtils.py
# @Author  : Mr.Favian096
# @Description : This file is used to provide tool functions for RNABindBERT
from collections import Counter
from ModelConfig import ModelConfig

import pandas as pd


def get_kmers(
        seq: str,
        k: int = ModelConfig.k
) -> list[str]:
    """
    return the string sequence of k-mers
    :param seq: string sequence
    :param k: the number of k-mers
    :return: divided list
    """
    return [seq[i:i + k] for i in range(0, len(seq) - k + 1)]


def generate_vocab_dict(
        train_file: str,
        seq_keys: list[str]
) -> int:
    """
    generate the vocabulary dictionary
    :param seq_keys: the key of each sequences
    :param train_file: base on train file
    :return: the vocabulary dictionary length
    """
    dataset: pd.DataFrame = pd.read_csv(train_file, sep='\t')
    seq_list: list = []

    for seq_key in seq_keys:
        seqs: list = dataset[seq_key].to_list()
        for seq in seqs:
            seq_list += get_kmers(seq)

    counter: list[tuple[str, int]] = sorted(Counter(seq_list).items(), key=lambda x: x[1], reverse=False)
    special_tokens: list[str] = ModelConfig.special_tokens
    seq_list: list[str] = [token for token, count in counter]
    vocabs: list[str] = special_tokens + seq_list

    with open(ModelConfig.vocab_file, 'w', encoding='utf-8') as file:
        for token in vocabs:
            file.write(token + '\n')

    return len(vocabs)
