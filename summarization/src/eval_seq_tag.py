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

def postprecoess(predict, batch):
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
        pred[batch['id'][i]] = sent
        
    return pred

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
    #output_file = Path(args.output_path)
    output_file = open(args.output_path, 'w')
    val_losses = []
    for batch in tqdm(eval_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            batch['text'] = batch['text'].cuda()
        val_predict = model(batch['text'], batch['text_len'])
        val_predict = torch.sigmoid(val_predict).round().squeeze()
        pred = postprecoess(val_predict,batch)
        pred_output = [json.dumps({'id':key, 'predict_sentence_index': [value]}) for key, value in pred.items()]
        output_file.write('\n'.join(pred_output))
        output_file.write('\n')
        
        '''
        print(len(pred))
        print(pred)
        print(val_predict[0])
        print(batch['id'][0])
        print(batch['label'][0])
        print(batch['sent_range'][0])
        '''
    output_file.close()
    
        
    
    
    


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data_path')
    parser.add_argument('output_path')
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