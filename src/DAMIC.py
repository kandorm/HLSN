import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.autograd import Variable

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from transformers import BertModel

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, attn_bias=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        if attn_bias is not None:
            attn = attn + attn_bias
        norm_attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(norm_attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Sequential(nn.Linear(d_model, n_head * d_k, bias=False), nn.ReLU())
        self.w_ks = nn.Sequential(nn.Linear(d_model, n_head * d_k, bias=False), nn.ReLU())
        self.w_vs = nn.Sequential(nn.Linear(d_model, n_head * d_v, bias=False), nn.ReLU())
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, attn_bias=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask, attn_bias=attn_bias)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        #q = self.dropout(q)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.w_1.weight)
        init.xavier_normal_(self.w_2.weight)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        #self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q_input, v_input, slf_attn_mask=None, attn_bias=None):
        enc_output, enc_slf_attn = self.slf_attn(q_input, v_input, v_input, mask=slf_attn_mask, attn_bias=attn_bias)
        #enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class LSTM_attention(nn.Module):
    def __init__(self, input_size, hidden_size, n_head, dk, dv, dropout=0.1):
        super(LSTM_attention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.droplstm = nn.Dropout(dropout)
        self.p_encoder = EncoderLayer(hidden_size*2, hidden_size*4, n_head, dk, dv, dropout)
        self.c_encoder = EncoderLayer(hidden_size*2, hidden_size*4, n_head, dk, dv, dropout)
        self.init_weight()

    def init_weight(self):
        for weights in [self.lstm.weight_hh_l0, self.lstm.weight_ih_l0, self.lstm.weight_ih_l0_reverse, self.lstm.weight_hh_l0_reverse]:
            init.orthogonal_(weights)

    def co_attention(self, A, B, attn_mask=None):
        attn = torch.matmul(A, B.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, B)
        return output, attn

    def forward(self, lstm_out, p_label, c_label, sen_embed, hidden=None, attn_mask=None, cor_mat=None, cor_lambda=None):
        batch_size, timesteps, _ = lstm_out.size()
        lstm_out, _ = self.lstm(lstm_out, hidden)
        lstm_out = self.droplstm(lstm_out)

        label_attn, p_att = self.p_encoder(lstm_out, p_label)
        # attn_bias = None
        # if cor_mat is not None and cor_lambda is not None:
        #     attn_bias = cor_lambda * torch.matmul(F.softmax(p_att.mean(1), dim=-1), cor_mat)
        label_attn, c_att = self.c_encoder(label_attn, c_label)

        sen_out, sen_att = self.co_attention(label_attn.view(batch_size*timesteps, 1, -1), sen_embed, attn_mask)
        sen_out = sen_out.view(batch_size, timesteps, -1)

        lstm_out = torch.cat((sen_out, lstm_out, label_attn), -1)
        return lstm_out, c_att, c_att, sen_att


class DAMIC(nn.Module):
    def __init__(self, hidden_size, output_size, output_size_P, weights_matrix, n_heads, stack_num, c_dropout, l_dropout, da_map_id):
        super(DAMIC, self).__init__()
        hidden_size = 768 // 2
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_size_P = output_size_P
        self.n_heads = n_heads
        self.stack_num = stack_num
        self.da_map_id = da_map_id

        self.cor_mat = nn.Parameter(self.create_correlation_matrix(da_map_id))
        self.cor_lambda = nn.Parameter(torch.ones(1))

        num_embeddings, embedding_dim = weights_matrix.size()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.load_state_dict({'weight': weights_matrix})
        #self.embedding.weight.requires_grad = False

        self.sen_rnn = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.sen_dropout = nn.Dropout(l_dropout)

        # pretrained_weights = './bert/'
        # self.context_embed = BertModel.from_pretrained(pretrained_weights)
        # input_size = 768
        # self.input_size = input_size
        input_size = hidden_size * 2
        self.cxt_rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        bi_output_size = hidden_size * 2
        self.cxt_dropout = nn.Dropout(l_dropout)

        self.c_lab = nn.Parameter(self.random_embedding_label(output_size, bi_output_size))
        self.p_lab = nn.Parameter(self.random_embedding_label(output_size_P, bi_output_size))

        self.c_encoder = EncoderLayer(bi_output_size, bi_output_size*2, n_heads, bi_output_size//n_heads, bi_output_size//n_heads, c_dropout)
        self.p_encoder = EncoderLayer(bi_output_size, bi_output_size*2, n_heads, bi_output_size//n_heads, bi_output_size//n_heads, c_dropout)

        self.lstm_attention_stack = nn.ModuleList([LSTM_attention(input_size+bi_output_size*2, hidden_size, n_heads, hidden_size*2//n_heads, hidden_size*2//n_heads, c_dropout) for i in range(stack_num-2)]
                                                + [LSTM_attention(input_size+bi_output_size*2, hidden_size, 1, hidden_size*2, hidden_size*2, c_dropout)])

        self.h2c = nn.Linear(input_size+bi_output_size*2, output_size)
        self.h2p = nn.Linear(input_size+bi_output_size*2, output_size_P)

        self.init_weight()

    def init_weight(self):
        for weights in [self.sen_rnn.weight_hh_l0, self.sen_rnn.weight_ih_l0, self.sen_rnn.weight_ih_l0_reverse, self.sen_rnn.weight_hh_l0_reverse]:
            init.orthogonal_(weights)
        for weights in [self.cxt_rnn.weight_hh_l0, self.cxt_rnn.weight_ih_l0, self.cxt_rnn.weight_ih_l0_reverse, self.cxt_rnn.weight_hh_l0_reverse]:
            init.orthogonal_(weights)

    def random_embedding_label(self, size, dim):
        out = np.zeros((size, dim))
        for i in range(size):
            out[i] = np.random.normal(scale=0.6, size=(dim, ))
        return torch.Tensor(out)

    def create_correlation_matrix(self, lab_map):
        out = np.zeros((self.output_size_P, self.output_size))
        for i in range(len(lab_map)):
            for j in lab_map[i]:
                out[i][j] = 1.0
        return torch.Tensor(out)

    def co_attention(self, A, B, attn_mask=None):
        attn = torch.matmul(A, B.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, B)
        return output, attn

    def forward(self, dialogue, seq_mask, trg_seqs):
        # print(dialogue.size())

        batch_size, timesteps, sent_len = dialogue.size()
        c_in = dialogue.view(batch_size*timesteps, sent_len) # (batch*timesteps, sent_len)
        embedded = self.embedding(c_in)  # (batch*timesteps, sent_len, embed)
        sent_length = torch.sum(seq_mask, dim=-1).view(-1) # (batch*timesteps)

        self.sen_rnn.flatten_parameters()
        embed_input_x_packed = nn.utils.rnn.pack_padded_sequence(embedded, sent_length, batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, _ = self.sen_rnn(embed_input_x_packed)   # None represents zero initial hidden state
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs_packed, batch_first=True)
        c_out = self.sen_dropout(encoder_outputs)
        c_out = c_out.view(batch_size, timesteps, -1, self.hidden_size*2)
        _, _, sent_len, _ = c_out.size()
        seq_mask = seq_mask[:, :, :sent_len].float()

        sen_att = seq_mask / seq_mask.sum(-1, keepdims=True) # (batch, timesteps, sent_len)
        sen_att = sen_att.view(batch_size, timesteps, 1, sent_len)
        r_in = torch.matmul(sen_att, c_out).squeeze(2) # (batch, timesteps, hidden*2)

        # c_out = self.context_embed(dialogue.view(-1, sent_len))[0]
        # seq_mask = seq_mask.float()
        # r_in = c_out[:, 0]
        # c_out = c_out.view(batch_size, timesteps, sent_len, self.input_size)
        # r_in = r_in.view(batch_size, timesteps, self.input_size)

        self.cxt_rnn.flatten_parameters()
        r_out, hidden = self.cxt_rnn(r_in) # (batch, timesteps, hidden*2)
        r_out = self.cxt_dropout(r_out)

        c_lab = torch.stack([self.c_lab]*batch_size, dim=0)
        p_lab = torch.stack([self.p_lab]*batch_size, dim=0)
        # cor_mat = torch.stack([self.cor_mat]*batch_size, dim=0)

        # p_att = F.softmax(p_att.mean(1), dim=-1)
        # attn_bias = self.cor_lambda * torch.matmul(p_att, self.cor_mat)
        lab_hidden, p_att = self.p_encoder(r_out, p_lab)
        lab_hidden, c_att = self.c_encoder(lab_hidden, c_lab)
        sen_embed = c_out.view(batch_size*timesteps, sent_len, -1)
        attn_mask = seq_mask.view(batch_size*timesteps, 1, sent_len)

        sen_out, sen_att = self.co_attention(lab_hidden.view(batch_size*timesteps, 1, -1), sen_embed, attn_mask)
        sen_out = sen_out.view(batch_size, timesteps, -1)

        lstm_out = torch.cat((sen_out, r_out, lab_hidden), dim=-1)

        for layer in self.lstm_attention_stack:
            lstm_out, p_att, c_att, sen_att = layer(lstm_out, p_lab, c_lab, sen_embed, hidden, attn_mask)

        lab_loss = Variable(torch.zeros(1)).to(device)
        for i in range(len(self.da_map_id)):
            for j in self.da_map_id[i]:
                lab_loss += torch.sum((self.p_lab[i] - self.c_lab[j])**2) / 2

        output = self.h2c(lstm_out)
        output_P = self.h2p(lstm_out)

        return output, output_P, lab_loss, lstm_out
