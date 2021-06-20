import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math, random, string, os, sys
from submodels.context_feature_extractor import CNN_Embedding
from DAMIC import DAMIC as basemodel

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_rnn(input_size, hidden_size, lstm_layers, l_dropout, gru):
    if gru:
        return nn.GRU(input_size, hidden_size, num_layers = lstm_layers, dropout = l_dropout, bidirectional = False, batch_first=True)
    else:
        return nn.LSTM(input_size, hidden_size, num_layers = lstm_layers, dropout = l_dropout, bidirectional = False, batch_first=True)

class DAMIC(nn.Module):
    def __init__(self, hidden_size, output_size, bi, weights_matrix, lstm_layers, n_filters, filter_sizes, c_dropout, l_dropout, teacher_forcing_ratio = None, gru = False, highway=False):
        super(DAMIC, self).__init__()
        self.highway = highway
        self.base = basemodel(hidden_size=1100, output_size=output_size, bi=True, weights_matrix=weights_matrix, lstm_layers=2, n_filters=200, filter_sizes=[3,4,5], c_dropout=0.4, l_dropout=0.2, teacher_forcing_ratio=None, gru=False, highway=False)
        # bmodel = bmodel.to(device)
        # bmodel = nn.DataParallel(bmodel)
        # original saved file with DataParallel
        state_dict = torch.load('./model/rnmoaknnpi/6')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        self.base.load_state_dict(new_state_dict)
        self.base.eval()

        # self.hidden_size = hidden_size
        self.output_size = output_size
        self.bi = bi
        self.teacher_forcing_ratio = teacher_forcing_ratio
    
        # self.fc = nn.Sequential(
        #     nn.Linear(len(filter_sizes)*n_filters, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        # )

        # self.fc = nn.Linear(len(filter_sizes)*n_filters, output_size)

        # self.e2e = nn.Sequential(
        #   nn.Linear(hidden_size, hidden_size),
        #   nn.ReLU(),
        #   nn.Dropout(p=0.2),
        # )
        
        input_size = output_size

        if bi:
            self.rnn = construct_rnn(input_size, hidden_size, lstm_layers, l_dropout, gru)
            bi_output_size = hidden_size * 2
        else:
            bi_output_size = hidden_size

        if self.highway:
            bi_output_size += input_size

        if self.teacher_forcing_ratio is not None:
            input_size += output_size
        self.rnn_r = construct_rnn(input_size, hidden_size, lstm_layers, l_dropout, gru)

        self.h2o = nn.Sequential(
            nn.Linear(bi_output_size, output_size),
            nn.Sigmoid(),
            # # MLP
            # nn.Linear(bi_output_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(hidden_size, output_size),
            # nn.Sigmoid(),
        )

    def forward(self, dialogue, targets = None):
        # print(dialogue.size())

        batch_size, timesteps, sent_len = dialogue.size()

        c_out = self.base(dialogue, targets)
        c_out = c_out.view(batch_size * timesteps, -1)
        thresholds = [0.48970118, 0.34025221, 0.17722994, 0.32379692, 0.1723266, 0.33252252, 0.26682911, 0.26431107, 0.2005045, 0.22233647, 0.50928269, 0.35311607]
        thresholds = torch.FloatTensor(thresholds).to(device)
        c_out = c_out > thresholds
        c_out = c_out.float()
        # print(c_out)
        
        r_in = c_out.view(batch_size, timesteps, -1)
        
        max_len = r_in.size()[1]
        r_out_vec = [None] * max_len
        predict_vec = [None] * max_len

        if self.bi:
            self.rnn.flatten_parameters()
            for i in range(max_len):
                i_r = max_len-i-1
                if i == 0:
                    r_out_step, h = self.rnn(r_in[:, i_r].unsqueeze(1))
                else:
                    r_out_step, h = self.rnn(r_in[:, i_r].unsqueeze(1), h)
                r_out_vec[i_r] = r_out_step
        
        self.rnn_r.flatten_parameters()
        for i in range(max_len):            
            # context input
            rnn_input = r_in[:, i].unsqueeze(1)
            
            # Scheduled Sampling
            if self.teacher_forcing_ratio is not None:
                
                if i == 0:
                    rnn_input = torch.cat([torch.empty(batch_size, 1, self.output_size, dtype=torch.float).fill_(.0).to(device), rnn_input], dim=2)
                elif self.teacher_forcing_ratio > 0 and random.random() < self.teacher_forcing_ratio:
                    # Teacher Forcing
                    assert targets is not None
                    rnn_input = torch.cat([targets[:, i-1].unsqueeze(1), rnn_input], dim=2)
                else:
                    rnn_input = torch.cat([predict_vec[i-1], rnn_input], dim=2)
            
            if i == 0:
                r_out_step, h = self.rnn_r(rnn_input)
            else:
                r_out_step, h = self.rnn_r(rnn_input, h)
            
            if self.bi:
                r_out_step = torch.cat((r_out_vec[i], r_out_step), dim=2)

            if self.highway:
                r_out_step = torch.cat((rnn_input, r_out_step), dim=2)
            
            predict_vec[i] = self.h2o(r_out_step)                
        
        predicts = torch.cat(predict_vec, dim=1)
        
        return predicts