import sys
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.param as param
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

class Attention_Decoder(nn.Module):
    def __init__(self, embedding, out_features):
        super(Attention_Decoder, self).__init__()
        print("\tInitializing LSTM Decoder with Attention Mechanism...")

        self.param = param.param()
        self.embedding = embedding
        self.embedding_input_size = embedding.embedding_dim
        self.embedding_layer_dropout = nn.Dropout(self.param.dropout)
        self.embedding_output_size = out_features

        if self.param.gru:
            self.decoder = nn.GRU(input_size=self.embedding_input_size, hidden_size=self.param.hidden_size, num_layers=self.param.num_layers, dropout=self.param.dropout)
        else:
            self.decoder = nn.LSTM(input_size=self.embedding_input_size, hidden_size=self.param.hidden_size, num_layers=self.param.num_layers, dropout=self.param.dropout)

        self.attention = nn.Linear(self.param.hidden_size, self.param.hidden_size)
        self.Whc = nn.Linear(in_features=2*self.param.hidden_size, out_features=self.param.hidden_size)
        self.Ws = nn.Linear(in_features=self.param.hidden_size, out_features=self.embedding_output_size)

    def forward(self, encoder_output, decoder_input, last_hidden):
        print("\t\t\tRunning Attention Decoder...")

        decoder_embedding = self.embedding_layer_dropout(self.embedding(decoder_input))
        decoder_embedding = decoder_embedding.view(1, decoder_input.size(0), self.embedding_input_size)

        decoder_output, decoder_hidden = self.decoder(decoder_embedding, last_hidden)
        alignment_vector = torch.sum(decoder_output*self.attention(encoder_output), dim=2)
        alignment_vector = F.softmax(alignment_vector.t(), dim=1).unsqueeze(1)

        context_vector = alignment_vector.bmm(encoder_output.transpose(0, 1))

        ReLU_input = torch.cat((decoder_output.squeeze(0), context_vector.squeeze(1)), 1)
        ReLU_output = F.relu(self.Whc(ReLU_input))

        softmax_output = F.log_softmax(self.Ws(ReLU_output), dim=1)

        return softmax_output, decoder_hidden