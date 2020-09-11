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
from utils import Tokenizer


def main():
    TRAIN = 'datasets/seq2seq/train.pkl'
    train = pickle.load(open(TRAIN, 'rb'))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=train.collate_fn)

    embedding_matrix = pickle.load(open("datasets/seq2seq/embedding.pkl", 'rb'))
    tokenizer = Tokenizer(lower=True)
    tokenizer.set_vocab(embedding_matrix.vocab)

    
    
    
    encoder = RNNEncoder(300, "datasets/seq2seq/embedding.pkl")
    decoder = RNNDecoder(300, "datasets/seq2seq/embedding.pkl")


    model = Seq2Seq(encoder, decoder, device).to(device)
    print(model)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index = 0)

    n_epochs = 6
    print_every = 2238
    counter = 0
    valid_loss_min = np.Inf
    model.train()
    for epoch in range(1, n_epochs+1):
        logging.info('Training')
        train_losses = []
        loss = 0
        counter = 0
        for batch in tqdm(train_loader):
            counter += 1
            #print("begining")
            #print(batch['text'].size())
            #[batch txt_len]
            #print(batch['summary'].size())
            #[batch trg_len]
            #print(len(batch['padding_len']))
            #[batch]
            optimizer.zero_grad()
            
            output = model(batch)
            #print("model output")
            #print(output.size())
            #[batch, trg_len, embedding word]
            output_dim = output.shape[-1]
            #print(output[:,0,:])
            output = output[:,1:,:].reshape(-1, output_dim)
            #print(output.size())
            #[batch*(trg_len -1), embedding word]
            target = batch['summary'][:,1:].reshape(-1).to(device)
            
            #print(target.size())
            #[batch*(trg_len-1)]
            loss = criterion(output, target) 
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            
            
        
        checkpoint_path = f'src/model_state/seq2seq/ckpt.{epoch}.pt'
        torch.save(
            {
                'state_dict' : model.state_dict(),
                'epoch': epoch,
            },
            checkpoint_path
        )
        print("Epoch: {}/{}...".format(epoch, n_epochs),
                                "Loss: {:.6f}...".format(np.mean(train_losses)))
        
            
    
if __name__ == '__main__':
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main()