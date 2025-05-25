# -*- coding: utf-8 -*-
# @File    : SeqBindBERTTune.py
# @Author  : Mr.Favian096
# @Description : This file is used to fine tune file form pre-train
import os

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from BERT.SeqBindBERT import SeqBindBERT
from BERT.SeqDataset import TuneDataset
from Config import Config

model = SeqBindBERT(bert=torch.load(
    f='./pre_train_' +
      str(Config.token_max_length) + '_' + str(Config.k_mers) + '.pth',
    weights_only=False)
)

tune_dataset_df = pd.read_csv(Config.train_dataset, sep='\t')
tune_data_list = [{'pre': row.miRNA_seq, 'post': row.LncRNA_seq, 'label': row.label}
                  for row in tune_dataset_df.itertuples()]

dataset = TuneDataset(tune_dataset=tune_data_list)

tune_dataloader = DataLoader(
    dataset,
    batch_size=Config.batch_size,
    num_workers=os.cpu_count(),
    shuffle=True
)

loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

loss_acc = {'loss': [], 'acc': []}
model.to(device=Config.DEVICE)
epochs = 2
for epoch in range(epochs):
    data_iter = tqdm(tune_dataloader, desc='Tuning')
    for data in data_iter:
        data = {k: v.to(device=Config.DEVICE) for k, v in data.items()}
        pred = model(data['token_ids'], segment_ids=data['segment_ids'])

        loss = loss_fun(pred, data['label'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = torch.argmax(pred, dim=1).eq(data['label']).float().mean().item()

        data_iter.set_postfix({'loss': loss.item(), 'acc': acc})
        loss_acc['loss'].append(loss.item())
        loss_acc['acc'].append(acc)

loss_acc_df = pd.DataFrame(loss_acc)
loss_acc_df.to_csv(
    './loss_acc_' + str(Config.token_max_length) + '_' + str(Config.k_mers) + '.csv',
    sep='\t',
    encoding='utf-8')
