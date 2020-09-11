import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from RNNEncoder import RNNEncoder
from RNNDecoder import RNNDecoder
from seq2seq import Seq2Seq
from dataset import Seq2SeqDataset
import logging
import os
import argparse
import json
from utils import Tokenizer
import collections

def eval(args):
    batch_size=32
    train_on_gpu = torch.cuda.is_available()
    
    enc = RNNEncoder(300, args.embedding_file)
    dec = RNNDecoder(300, args.embedding_file)
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
 
    model = Seq2Seq(enc, dec, device).to(device)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['state_dict'])
    
        
    model.eval()

    embedding_matrix = pickle.load(open(args.embedding_file, 'rb'))
    tokenizer = Tokenizer(lower=True)
    tokenizer.set_vocab(embedding_matrix.vocab)
    eval_data = pickle.load(open(args.test_data_path, 'rb'))
    eval_loader = DataLoader(eval_data, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=eval_data.collate_fn)
    
    
  
    output_file = open(args.output_path, 'w')
    val_losses = []
    prediction={}
    for batch in tqdm(eval_loader):
        pred = model(batch,0) 
        pred = torch.argmax(pred, dim=2)
        # batch, seq_len
        
        for i in range(len(pred)):
            prediction[batch['id'][i]] = tokenizer.decode(pred[i]).split('</s>')[0].split(' ',1)[1]
    pred_output = [json.dumps({'id':key, 'predict': value}) for key, value in sorted(prediction.items(), key=lambda item: item[0])]
    output_file.write('\n'.join(pred_output))
    output_file.write('\n')
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