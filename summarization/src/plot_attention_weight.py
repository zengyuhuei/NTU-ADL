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
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib

matplotlib.use('Agg')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(10,5), dpi = 1000)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().detach().numpy(), vmin=0, vmax=1,cmap='bone', aspect='auto')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    
    plt.savefig('attention_weight.png')
    #exit()
    #plt.show()

def main(args):
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
    eval_data = pickle.load(open(args.test_data_path, 'rb'))
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, collate_fn=eval_data.collate_fn)

    val_losses = []
    prediction={}
    for batch in tqdm(eval_loader):
        text = batch['text'].to(device)
        text_len = batch['padding_len']
        truth = batch["summary"].to(device)

        text = text.permute(1, 0)
        truth = truth.permute(1, 0)
        #print(text.size())
        pred ,attn= model(text, text_len, truth, 0)
        #print(pred.size())
        pred = torch.argmax(pred, dim=2)
        #print(pred.size())
        pred = pred.permute(1, 0)
        #print(pred.size())
        break
    
    text = text.permute(1, 0)
    
    
    
    attn = attn.permute(1, 0, 2)
    for i in range(len(text[-1])):
        if text[-1][i] == 0:
            text_stop = i
            break
    for i in range(len(pred[-1])):
        if pred[-1][i] == 2:
            pred_stop = i
            break
    input = text[-1][0:text_stop]
    attention = attn[-1][1:pred_stop+1,0:text_stop]
    output = pred[-1][1:pred_stop+1]
    showAttention([embedding.vocab[t] for t in input], [embedding.vocab[t] for t in output], attention)
  
    
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
    main(args)
