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

def train():
    TRAIN = '../datasets/seq_tag/train.pkl'
    VALID = '../datasets/seq_tag/valid.pkl'    
    train = pickle.load(open(TRAIN, 'rb'))
    valid = pickle.load(open(VALID, 'rb'))


    batch_size=128
    train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=train.collate_fn)
    valid_loader = DataLoader(valid, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=valid.collate_fn)

    
    train_on_gpu = torch.cuda.is_available()

    #train model
    model = RNN(batch_size, 300)
    pos_weight = torch.tensor([7])
    if train_on_gpu:
        model.cuda()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight =  torch.tensor([7]).cuda())
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight =  torch.tensor([7]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    valid_loss_min = np.Inf
    print(model)

    
    n_epochs = 10
    print_every = 559*2
    counter = 0
    model.train()
    for epoch in range(1, n_epochs+1):
        logging.info('Training') 
        for batch in tqdm(train_loader):
            counter += 1
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                batch['text'], batch['label'] = batch['text'].cuda(), batch['label'].cuda()
            optimizer.zero_grad()
            train_predict = model(batch['text'], batch['text_len']) 
            # clear the gradients of all optimized variables
            
            # calculate the batch loss
            loss = criterion(train_predict.squeeze(), batch['label'].float())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), clip)
            # perform a single optimization step (parameter update)
            optimizer.step()
            
 
            if counter%print_every == 0:
                logging.info('Validating') 
                val_losses = []
                model.eval()
                for batch in tqdm(valid_loader):
                    # move tensors to GPU if CUDA is available
                    if train_on_gpu:
                        batch['text'], batch['label'] = batch['text'].cuda(), batch['label'].cuda()
                    val_predict = model(batch['text'], batch['text_len'])
                    val_loss = criterion(val_predict.squeeze(), batch['label'].float())
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(epoch, n_epochs),
                        "Step: {}...".format(counter),
                        "Loss: {:.6f}...".format(loss.item()),
                        "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    checkpoint_path = f'./model_state/seq_tag/ckpt.{epoch}.pt'
                    torch.save(
                        {
                            'state_dict' : model.state_dict(),
                            'epoch': epoch,
                        },
                        checkpoint_path
                    )
                    logging.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                    valid_loss_min = np.mean(val_losses)
    

if __name__ == '__main__':
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    train()