import torch.nn as nn
import torch
import pickle
import torch.nn.utils.rnn as rnn_utils
class RNN(nn.Module):
    def __init__(self, batch_size, word_vector, embedding_matrix_path):
        super(RNN, self).__init__()

        embedding_matrix = pickle.load(open(embedding_matrix_path, 'rb'))
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix.vectors))
        self.dropout1 = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            input_size  = word_vector,
            hidden_size = 128,
            dropout=0.5,
            num_layers =  2,
            batch_first = True,
            bidirectional  = True,
        )
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Sequential(nn.Linear(128*2, 1))

    def forward(self, text, text_len):
        embed_text = self.embedding(text)
        embed_text = self.dropout1(embed_text)
        pack_padded_text = rnn_utils.pack_padded_sequence(embed_text, text_len, batch_first=True)
        out , hidden= self.lstm(pack_padded_text, None)   # None represents zero initial hidden state
        out_pad_packed, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
        output = self.dropout2(out_pad_packed)
        output = self.linear(output)
        return output
    '''
    def initHidden(self, batch_size):
        weight = next(self.parameters()).data
        #2*2 because bidirectional  = True and num_layer = 2
        hidden = (weight.new(2*2, batch_size, 128).zero_().cuda(),
                      weight.new(2*2, batch_size, 128).zero_().cuda())
        return hidden
    '''