import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math, random, string, os, sys
from submodels.context_feature_extractor import CNN_Embedding

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DAMIC(nn.Module):
    def __init__(self, hidden_size, output_size, bi, weights_matrix, lstm_layers, n_filters, filter_sizes, c_dropout, l_dropout, teacher_forcing_ratio = None):
        super(DAMIC, self).__init__()

        # self.hidden_size = hidden_size
        self.output_size = output_size
        # self.bi = bi
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.context_feature_extractor = CNN_Embedding(weights_matrix, n_filters, filter_sizes, c_dropout)
        
        input_size = len(filter_sizes)*n_filters

        self.rnn = nn.LSTM(output_size, hidden_size, num_layers = lstm_layers, dropout = l_dropout, bidirectional = False, batch_first=True)

        self.h2o = nn.Sequential(
            nn.Linear(hidden_size+input_size, output_size),
            nn.Sigmoid(),
            # MLP
            # nn.Linear(hidden_size+input_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(hidden_size, output_size),
            # nn.Sigmoid(),
        )

    def forward(self, dialogue, targets = None):

        batch_size, timesteps, sent_len = dialogue.size()
        
        c_out = self.context_feature_extractor(dialogue)

        #c_out = [batch size * timesteps, n_filters * len(filter_sizes)]

        r_in = c_out.view(batch_size, timesteps, -1)
        
        max_len = r_in.size()[1]
        predict_vec = [None] * max_len
        
        self.rnn.flatten_parameters()
        for i in range(max_len):            
            # context input
            mlp_input = r_in[:, i].unsqueeze(1)
            
            # Scheduled Sampling
            if i == 0:
                rnn_input = torch.empty(batch_size, 1, self.output_size, dtype=torch.float).fill_(.0).to(device)
            elif self.teacher_forcing_ratio > 0 and random.random() < self.teacher_forcing_ratio:
                # Teacher Forcing
                assert targets is not None
                rnn_input = targets[:, i-1].unsqueeze(1)
            else:
                rnn_input = predict_vec[i-1]
            
            if i == 0:
                r_out_step, (h_n, h_c) = self.rnn(rnn_input)
            else:
                r_out_step, (h_n, h_c) = self.rnn(rnn_input, (h_n, h_c))
            
            r_out_step = torch.cat([mlp_input, r_out_step], dim=2)
            
            predict_vec[i] = self.h2o(r_out_step)                
        
        predicts = torch.cat(predict_vec, dim=1)
        
        return predicts