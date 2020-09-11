import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from RNNModel import RNN
from dataset import SeqTaggingDataset
import logging
import os
import argparse
import json
from matplotlib import pyplot as plt

def store(predict, batch, data):
    pred = {}
    for i in range(len(predict)):
        total = []
        max = -1
        sent = -1
        j = 0
       
        for start, end in batch['sent_range'][i]:
            if start < 300:
                if end > 300:
                    end = 300
                include = sum(predict[i][start:end])
                if include > max:
                    max = include
                    sent = j
            j = j + 1 
        
        if round(sent / len(batch['sent_range'][i]), 2) in data:
            data[round(sent / len(batch['sent_range'][i]), 2)] += 1
        else:
            data[round(sent / len(batch['sent_range'][i]), 2)] = 1

def eval(args):
    batch_size=128
    train_on_gpu = torch.cuda.is_available()
    model = RNN(batch_size, 300, args.embedding_file)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['state_dict'])
    if train_on_gpu:
        model.cuda()
    model.eval()

    
    eval_data = pickle.load(open(args.test_data_path, 'rb'))
    eval_loader = DataLoader(eval_data, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=eval_data.collate_fn)

    prediction = []
    val_losses = []
    data = {}
    dataset = {}
    for batch in tqdm(eval_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            batch['text'] = batch['text'].cuda()
        val_predict = model(batch['text'], batch['text_len'])
        val_predict = torch.sigmoid(val_predict).round().squeeze()
        store(val_predict,batch,data)
    
    dataset  = {i[0]:i[1] for i in sorted(data.items(), key=lambda kv: kv[0])}
    print(dataset)
    x = []
    y = []
    total = 0
    for key , value in dataset.items():
        x.append(key)
        y.append(value)
        total += value
    y = [i/total for i in y]
    plt.figure(figsize=(9,6))
    plt.bar(x, y, width = 0.01, facecolor = 'lightskyblue',edgecolor = 'white',)
    plt.ylabel('Density')
    plt.xlabel('Relative Location')
    plt.xlim(-0.05, 1) 
    plt.show()
    
    


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data_path')
    parser.add_argument('embedding_file')
    parser.add_argument('model_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    eval(args)