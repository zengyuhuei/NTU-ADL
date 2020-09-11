import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import Seq2SeqDataset
from attention_models import Encoder, Decoder, Seq2Seq, Attention
import time
import math
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    #for i, batch in enumerate(iterator):
    for batch in tqdm(iterator):
        truth = batch['summary'].to(device)
        truth = truth.permute(1, 0)
        optimizer.zero_grad()
        output, attention = model(batch)

        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        truth = truth[1:].reshape(-1)

        loss = criterion(output, truth)
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)




def main():
    # CONFIG
    DATASET_PATH = 'datasets/attention'
    DATASET_TYPES = ['train', 'valid', 'embedding']

    BATCH_SIZE = 32
    ENC_HID_DIM = 128
    DEC_HID_DIM = 128
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    PADDING_INDEX = 0

    # Load datsets and embedding from pickle
    datasets = {}

    for t in DATASET_TYPES:
        with open('{}/{}.pkl'.format(DATASET_PATH, t), 'rb') as f:
            datasets[t] = pickle.load(f)

    train_dataset = datasets['train']
    valid_dataset = datasets['valid']
    #test_dataset = datasets['test']
    embedding = datasets['embedding']
    embedding_matrix = embedding.vectors.to(device)

    # parameter from dataset

    #output_dim = datasets['train'].max_summary_len
    output_dim = len(embedding.vectors)
    embedding_dim = 300

    #print(input_dim, output_dim, embedding_dim)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    encoder = Encoder(embedding_dim, ENC_HID_DIM, DEC_HID_DIM,
                      embedding_matrix, N_LAYERS, ENC_DROPOUT)
    decoder = Decoder(output_dim, embedding_dim,
                      ENC_HID_DIM, DEC_HID_DIM, embedding_matrix, N_LAYERS, DEC_DROPOUT, attn)

    model = Seq2Seq(encoder, decoder, PADDING_INDEX, device).to(device)


    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_INDEX)

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True, collate_fn=train_dataset.collate_fn)
    valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle = False,collate_fn=valid_dataset.collate_fn)

    n_epochs = 30

    best_valid_loss = float('inf')

    for epoch in range(1,n_epochs+1):
        
        train_loss = train(model, train_iterator, optimizer, criterion)

        
        checkpoint_path = f'src/model_state/attention/ckpt.{epoch}.pt'
        torch.save(
            {
                'state_dict' : model.state_dict(),
                'epoch': epoch,
            },
            checkpoint_path
        )
        print("Epoch: {}/{}...".format(epoch, n_epochs),
                                "Loss: {:.6f}...".format(train_loss))


if __name__ == "__main__":
    main()
