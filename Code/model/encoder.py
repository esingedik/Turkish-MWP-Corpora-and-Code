import sys
from os import path
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils.param as param
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        print("\tInitializing Bidirectional Encoder...")

        self.param = param.param()
        self.device = device

        if self.param.gru:
            self.encoder = nn.GRU(input_size=self.param.input_embedding_size, hidden_size=self.param.hidden_size, num_layers=self.param.num_layers, dropout=self.param.dropout, bidirectional=True)
        else:
            self.encoder = nn.LSTM(input_size=self.param.input_embedding_size, hidden_size=self.param.hidden_size, num_layers=self.param.num_layers, dropout=self.param.dropout, bidirectional=True)

    def init_hidden(self):
        hidden = Variable(torch.zeros(2*self.param.num_layers, self.param.batch_size, self.param.hidden_size))
        hidden = hidden.to(self.device)
        return hidden

    def forward(self, sorted_question_tensor, sorted_question_tokens_length, question_token_indexes):
        print("\t\t\tRunning Bidirectional Encoder...")

        pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(sorted_question_tensor, sorted_question_tokens_length)
        encoder_output, encoder_hidden = self.encoder(pack_padded_sequence)
        encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_output)
        encoder_output = torch.index_select(input=encoder_output, dim=1, index=question_token_indexes)
        encoder_output = encoder_output[:, :, :self.param.hidden_size] + encoder_output[:, :, self.param.hidden_size:]

        return encoder_output, encoder_hidden