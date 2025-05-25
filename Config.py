# -*- coding: utf-8 -*-
# @File    : Config.py
# @Author  : Mr.Favian096
# @Description : Project data configure

import torch


class Config:
    batch_size = 32

    token_max_length = 512

    k_mers = 3

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CURRENT_PATH = './'

    train_dataset = './Datasets/train_datasetes.csv'

    figure_size = (9, 6.75)

    models_download_path = './Models/'
