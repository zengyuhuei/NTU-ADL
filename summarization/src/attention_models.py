import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from dataset import Seq2SeqDataset
import random


#torch.backends.cudnn.enabled = False

class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, embedding_matrix, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        #self.embedding.weight.requires_grad = False
        self.hid_dim = enc_hid_dim
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, n_layers,
                          bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        text_len = encoder_outputs.shape[0]

        # repeat decoder hidden state text_len times
        # hidden = [batch size, dec hid dim]
        hidden = hidden.unsqueeze(1).repeat(1, text_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(mask == True, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, embedding_matrix, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        #self.embedding.weight.requires_grad = False
        self.hid_dim = dec_hid_dim
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, n_layers)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

    def forward(self, text, hidden, encoder_outputs, mask):
        text = text.unsqueeze(0)
        embedded = self.embedding(text)
        attn = self.attention(hidden, encoder_outputs, mask)
        attn = attn.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(attn, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), attn.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, text_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.text_pad_idx = text_pad_idx
        self.device = device

    def create_mask(self, text):
        mask = (text != self.text_pad_idx).permute(1, 0)
        return mask

    def forward(self, batch, teacher_force_rate=0.5):
        data = batch['text'].to(self.device)
        data = data.permute(1, 0)
        batch_size = data.shape[1]

        if teacher_force_rate:
            truth = batch['summary'].to(self.device)
            truth = truth.permute(1, 0)
            truth_len = truth.shape[0]
            s_text = truth[0, :]
        else:
            truth_len = 40
            s_text = torch.tensor(batch_size*[1]).to(self.device)

        truth_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(truth_len, batch_size,
                              truth_vocab_size).to(self.device)
        attentions = torch.zeros(truth_len, batch_size,
                              data.shape[0]).to(self.device)
        encoder_outputs, hidden = self.encoder(data, batch['padding_len'])

         
        mask = self.create_mask(data)
        for i in range(1, truth_len):
            output, hidden, attention = self.decoder(
                s_text, hidden, encoder_outputs, mask)
            outputs[i] = output
            attentions[i] = attention
            teacher_force = random.random() < teacher_force_rate
            top1 = output.argmax(1)
            s_text = truth[i] if teacher_force else top1
        return outputs, attentions
