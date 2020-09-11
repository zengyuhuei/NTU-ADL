import torch.nn as nn
import torch
import pickle
import torch.nn.utils.rnn as rnn_utils
import random
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers  == decoder.num_layers , \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, batch, teacher_forcing_ratio = 0.5):
        
        data = batch['text'].to(self.device)
        batch_size = data.shape[0]
        
        if teacher_forcing_ratio:
            target = batch['summary'].to(self.device)
            trg_len = batch['summary'].shape[1]
            text = target[:,0]
        else:
            trg_len = 40
            text = torch.tensor(batch_size*[1]).to(self.device)
        trg_vocab_size = self.decoder.output_dim
        #print("seq2seq")
        #print(data.size())
        #[batch, txt_len]
        #print(target.size())
        #[batch, trg_len]
        #print(text.size())
        #[batch]
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        #print(outputs.size())
        #[batch, trg_len, #num of embedding_word]
        #hidden, cell = self.encoder(data, batch['padding_len'])
        hidden = self.encoder(data, batch['padding_len'])
        #print(hidden.size(), cell.size())
        #[n_direction*n_layer ,batch ,hidden_size] [n_direction*n_layer ,batch ,hidden_size] 
        
        
        for t in range(1, trg_len):
            #output, hidden, cell = self.decoder(text, hidden, cell)
            output, hidden = self.decoder(text, hidden)
            #print("after decoder")
            #print(output.size(), hidden.size(), cell.size())
            #[batch, n_direction*n_layer, embedding word][n_direction*n_layer ,batch ,hidden_size] [n_direction*n_layer ,batch ,hidden_size] 
            output = output.squeeze(1)
            #print(output.size())
            #[batch, embedding word]
            #print(output)
            outputs[:,t,:] = output
            #print(outputs)
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1) 
            #print(top1.size())
            #[batch]
            #print(target[:,t].size())
            #[batch]
            text = target[:,t] if teacher_force else top1
        return outputs
      
        