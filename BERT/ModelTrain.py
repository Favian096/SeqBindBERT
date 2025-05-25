# -*- coding: utf-8 -*-
# @File    : ModelTrain.py
# @Author  : Mr.Favian096
# @Description : This file is used to complete RNABindBERT model's pre-train

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd

from BERT import BERTPreTrain, BERT
from tqdm import tqdm

from ModelConfig import ModelConfig


class ScheduledOptim:
    def __init__(self, optimizer, model_dim, warmup_steps):
        self.optimizer: torch.optim = optimizer
        self.warmup_steps: int = warmup_steps
        self.current_steps: int = 0
        self.init_lr: float = np.power(model_dim, -0.5)

    def step_and_update_lr(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_steps])

    def update_learning_rate(self):
        self.current_steps += 1
        lr: float = self.init_lr * self.get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class BERTTrainer:
    def __init__(
            self,
            bert: BERT,
            vocab_size: int,
            train_dataloader: DataLoader,
            betas=(0.9, 0.999),
            weight_decay: float = 0.01,
            warmup_steps: int = 1000,
    ):

        self.device: torch.device = ModelConfig.device
        self.bert: BERT = bert
        self.model: BERTPreTrain = BERTPreTrain(bert, vocab_size).to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.train_data: DataLoader = train_dataloader

        self.optim: torch.optim = Adam(
            self.model.parameters(),
            lr=ModelConfig.lr,
            betas=betas,
            weight_decay=weight_decay
        )
        self.optim_schedule: ScheduledOptim = ScheduledOptim(
            self.optim,
            self.bert.hidden,
            warmup_steps=warmup_steps
        )

        self.mlm_loss_fun: nn.NLLLoss = nn.NLLLoss(ignore_index=0)
        self.nsp_loss_fun: nn.NLLLoss = nn.NLLLoss()

    def train(self, epoch: int):
        self.iteration(epoch, self.train_data)

    def iteration(
            self,
            epoch: int,
            data_loader: DataLoader,
    ):

        loss_acc_dict = {
            'loss': [],
            'acc': []
        }

        data_iter = tqdm(
            iterable=data_loader,
            desc=f'Training epoch {epoch + 1}',
        )

        for data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            nsp_output, mlm_output = self.model.forward(
                data['token_ids'],
                data['segment_ids'])

            next_loss: torch.Tensor = self.nsp_loss_fun(nsp_output, data['is_next'])

            mask_loss: torch.Tensor = self.mlm_loss_fun(mlm_output.transpose(1, 2), data['mlm_labels'])

            loss = next_loss + mask_loss

            self.optim_schedule.zero_grad()
            loss.backward()
            self.optim_schedule.step_and_update_lr()

            acc = nsp_output.argmax(dim=-1).eq(data["is_next"]).float().mean().item()

            post_fix = {
                'loss': loss.item(),
                'acc': acc
            }

            data_iter.set_postfix(post_fix)

            loss_acc_dict['loss'].append(loss.item())
            loss_acc_dict['acc'].append(acc)
        df = pd.DataFrame(loss_acc_dict)
        df.to_csv(
            ModelConfig.output_path + 'loss_acc_' + str(ModelConfig.token_num) + '_' + str(ModelConfig.k) + '.csv',
            sep='\t',
            mode='a',
            header=False,
            encoding='utf-8'
        )

    def save(self, file_path: str) -> None:
        torch.save(self.bert.cpu(), file_path)
        self.bert.to(self.device)
