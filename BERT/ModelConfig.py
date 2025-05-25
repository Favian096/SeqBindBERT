# -*- coding: utf-8 -*-
# @File    : ModelConfig.py
# @Author  : Mr.Favian096
# @Description : This file is used to configure RNABindBERT model
from Config import Config

class ModelConfig:
    device = Config.DEVICE

    output_path = Config.pre_train_output_path

    vocab_file = './vocab.txt'

    k = Config.k_mers

    batch_size = Config.batch_size

    nsp_reverse = False

    token_num = Config.token_max_length

    PAD = '[PAD]'
    pad_index = 0
    UNK = '[UNK]'
    unk_index = 1
    CLS = '[CLS]'
    cls_index = 2
    SEP = '[SEP]'
    sep_index = 3
    MASK = '[MASK]'
    mask_index = 4
    special_tokens = [PAD, UNK, CLS, SEP, MASK]

    embedding_dropout = 0.1

    encoder_dropout = 0.1

    feed_forward_dropout = 0.1

    model_hidden = 768

    embedding_size = model_hidden

    lr = 1e-4

    encoder_num = 12

    attention_head_num = 12
