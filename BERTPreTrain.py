# -*- coding: utf-8 -*-
# @File    : BERTPreTrain.py
# @Author  : Mr.Favian096
# @Description : This file is used to pre-train BERT model
import os

import pandas as pd
from torch.utils.data import DataLoader
from BERT.BERT import BERTPreTrain, BERT
from BERT.ModelTrain import BERTTrainer
from BERT.ModelUtils import generate_vocab_dict
from BERT.SeqDataset import PreTrainDataset
from Config import Config

vocabs_length = generate_vocab_dict(train_file=Config.train_dataset, seq_keys=['miRNA_seq', 'LncRNA_seq'])

train_dataset_df = pd.read_csv(Config.train_dataset, sep='\t')
train_data_list = [{'pre': row.miRNA_seq, 'post': row.LncRNA_seq, 'label': row.label}
                   for row in train_dataset_df.itertuples()]

dataset = PreTrainDataset(train_dataset=train_data_list)

train_dataloader = DataLoader(
    dataset,
    batch_size=Config.batch_size,
    num_workers=os.cpu_count(),
    shuffle=True
)

epochs = 2
trainer = BERTTrainer(
    bert=BERT(vocab_size=vocabs_length),
    vocab_size=vocabs_length,
    train_dataloader=train_dataloader,
    warmup_steps=epochs * len(train_dataloader),
)

for epoch in range(epochs):
    trainer.train(epoch)
    trainer.save(
        './pre_train_' + str(Config.token_max_length) + '_' + str(Config.k_mers) + '.pth')
