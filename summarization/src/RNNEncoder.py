import torch.nn as nn
import torch
import pickle
import torch.nn.utils.rnn as rnn_utils
class RNNEncoder(nn.Module): #, emb_dim, hid_dim, n_layers, dropout
    def __init__(self, word_vector, embedding_matrix_path):
        super().__init__()
        
        embedding_matrix = pickle.load(open(embedding_matrix_path, 'rb'))
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix.vectors))
        self.hid_dim = 128
        self.num_layers = 1
        self.lstm = nn.GRU(
            input_size  = word_vector,
            hidden_size = self.hid_dim,
            #dropout=0.5,
            num_layers =  self.num_layers,
            batch_first = True,
            bidirectional  = True,
        )
        
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Sequential(nn.Linear(self.hid_dim*2, 128), nn.Tanh())
        self.linear2 = nn.Sequential(nn.Linear(self.hid_dim*2, 128), nn.Tanh())
        
    def forward(self, text, padding_len):
        
        #print("encoder")
        embedded = self.embedding(text)
        #embedded = self.dropout(embedded)
        #print(embedded)
        #print(embedded.size())
        #[batch, text_len, embedding vector]
        #print(text)
        pack_padded_text = rnn_utils.pack_padded_sequence(embedded, padding_len, batch_first=True)
        #print(padding_len)
        #print(pack_padded_text.data)
        #print(pack_padded_text.data.size())
        packed_outputs, hidden = self.lstm(pack_padded_text)
        #print("lstm")
        #print(hidden.size(), cell.size())
        #[n_direction*n_layer ,batch, hidden_size] [n_direction*n_layer ,batch, hidden_size] 
        outputs, output_lengths = rnn_utils.pad_packed_sequence(packed_outputs ,batch_first=True)
        #print(outputs.size())

        hidden = torch.cat((hidden[0],hidden[1]), dim = 1).unsqueeze(0)
        #cell = torch.cat((cell[0],cell[1]), dim = 1).unsqueeze(1)
        #print(hidden.size(), cell.size())
        #[batch , 1 , n_direction*n_layer*hidden_size] [batch,1, n_direction*n_layer*hidden_size] 
        hidden = self.linear1(hidden)
        #cell = self.linear2(cell)
        #print("linear")
        #print(hidden.size(), cell.size())
        #print("permute")
        #hidden = hidden.permute(1,0,2) 
        #cell = cell.permute(1,0,2) 
        #print(hidden.size(), cell.size())
        #[n_direction*n_layer, batch ,hidden_size] [n_direction*n_layer, batch ,hidden_size]
        return hidden
        
