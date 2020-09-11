import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from attention_models import Encoder, Decoder, Seq2Seq, Attention
from dataset import Seq2SeqDataset
import logging
import os
import argparse
import json
from utils import Tokenizer
#python src/eval_attention.py datasets/attention/valid.pkl predict_attention_10.jsonl datasets/attention/embedding.pkl src/model_state/attention/ckpt.10.pt
#python scripts/scorer_abstractive.py predict_attention.jsonl datasets/attention/valid.pkl
#python src/preprocess_seq2seq_test.py data/valid.jsonl datasets/attention/valid.pkl datasets/attention/embedding.pkl
def eval(args):
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print(device)
    BATCH_SIZE = 32

    ENC_HID_DIM = 128
    DEC_HID_DIM = 128
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    PADDING_INDEX = 0

    embedding = pickle.load(open(args.embedding_file, 'rb'))
    tokenizer = Tokenizer(lower=True)
    tokenizer.set_vocab(embedding.vocab)
    embedding_matrix = embedding.vectors.to(device)

    output_dim = len(embedding.vectors)
    embedding_dim = 300


    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    encoder = Encoder(embedding_dim, ENC_HID_DIM, DEC_HID_DIM,
                      embedding_matrix, N_LAYERS, ENC_DROPOUT)
    decoder = Decoder(output_dim, embedding_dim,
                      ENC_HID_DIM, DEC_HID_DIM, embedding_matrix, N_LAYERS, DEC_DROPOUT, attn)

    model = Seq2Seq(encoder, decoder, PADDING_INDEX, device).to(device)


    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_INDEX)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['state_dict'])
    
        
    model.eval()

    
    eval_data = pickle.load(open(args.test_data_path, 'rb'))
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, collate_fn=eval_data.collate_fn)
    
    
  
    output_file = open(args.output_path, 'w')
    val_losses = []
    prediction={}
    for batch in tqdm(eval_loader):
        #print(text.size())
        pred ,attention= model(batch, 0)
        #print(pred.size())
        pred = torch.argmax(pred, dim=2)
        #print(pred.size())
        pred = pred.permute(1, 0)
        #print(pred.size())
        
   
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