import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from RNNModel import RNN
import time
from dataset import SeqTaggingDataset


TRAIN = 'datasets/seq_tag/train.pkl'
VALID = 'datasets/seq_tag/valid.pkl'    
train = pickle.load(open(TRAIN, 'rb'))
valid = pickle.load(open(VALID, 'rb'))




    
batch_size=128
train_loader = DataLoader(train, batch_size=batch_size, num_workers=0,\
                                                shuffle=True,collate_fn=collate_fn)
valid_loader = DataLoader(valid, batch_size=batch_size,  num_workers=0,\
                                                shuffle=False,collate_fn=collate_fn)


batch_data, batch_label, batch_len = iter(train_loader).next()


model = RNN(batch_size, 300)
model.cuda()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight =  torch.tensor([7]).cuda())                    
threshold = torch.tensor([0.5])
n_epochs = 10
counter = 0
valid_loss_min = np.Inf # track change in validation loss
model.train()
start_time = time.time()
for epoch in range(1, n_epochs+1):
    # keep track of training and validation loss
    train_acc = 0
    val_acc = 0
    sigmoid = nn.Sigmoid()
    print("### Training ###")
    clip = 5
    print_every = 559*2

    for batch_data, batch_label, batch_len in tqdm(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            batch_data, batch_label, batch_len = torch.Tensor(batch_data).long().cuda(), torch.Tensor(batch_label).cuda(), torch.Tensor(batch_len).cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        counter += 1
        train_predicted = model(batch_data, batch_len) 
        predict = sigmoid(train_predicted).round()
        # calculate the batch loss
        #predict.requires_grad_()
        loss = criterion(train_predicted.squeeze(), batch_label.float())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        # perform a single optimization step (parameter update)
        optimizer.step()
        

        
        if counter%print_every == 0:
            print("### Validating ###")
            val_losses = []
            model.eval()
            for batch_data, batch_label, batch_len in tqdm(valid_loader):
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    batch_data, batch_label, batch_len = torch.Tensor(batch_data).long().cuda(), torch.Tensor(batch_label).cuda(), torch.Tensor(batch_len).long().cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
            
                val_predicted = model(batch_data, batch_len)  
                predict = sigmoid(val_predicted).round()
                # calculate the batch loss
                #predict.requires_grad_()
                val_loss = criterion(val_predicted.squeeze(), batch_label.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(epoch, n_epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), rnn_pth)
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)








