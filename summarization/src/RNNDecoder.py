import torch.nn as nn
import torch
import pickle
import torch.nn.utils.rnn as rnn_utils
class RNNDecoder(nn.Module): #,  emb_dim, hid_dim, n_layers, dropout
    def __init__(self, word_vector, embedding_matrix_path):
        super().__init__()
        
        
        embedding_matrix = pickle.load(open(embedding_matrix_path, 'rb'))
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix.vectors))
        self.hid_dim = 128
        self.num_layers = 1
        self.output_dim = embedding_matrix.vectors.size(0)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.GRU(
            input_size  = word_vector,
            hidden_size = self.hid_dim,
            #dropout=0.5,
            num_layers =  self.num_layers,
            batch_first = True,
        )
        self.linear = nn.Sequential(nn.Linear(self.hid_dim, self.output_dim))
        
    def forward(self, text, hidden):
        
        #print("decoder")
        #print(text.size(), hidden.size(), cell.size())
        #[batch] [n_direction*n_layer ,batch ,hidden_size] [n_direction*n_layer ,batch ,hidden_size]
        text = text.unsqueeze(-1)
        #print("text squeezing")
        #print(text.size())
        #[batch size, 1]
        embedded = self.embedding(text)
        #print(embedded.size())
        #[batch size, 1, emb dim]
        # embedded = self.dropout(self.embedding(text))
        # output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output, hidden = self.lstm(embedded, hidden)
        #print("lstm")
        #print(output.size(), hidden.size(), cell.size())
        #[batch size, 1, hid dim] [1, batch size, hid dim] [1, batch size, hid dim]
        prediction = self.linear(output)
        #print("after linear")
        #print(prediction.size()) 
        #[batch size, 1, output dim]
        
        return prediction, hidden
   