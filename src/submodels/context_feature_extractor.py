import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math, random, string, os, sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from submodels.dynamic_k_max import DynamicKMaxPooling

# def kmax_pooling(x, dim, k):
#     # print(x.shape)
#     index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
#     return x.gather(dim, index)

torch.manual_seed(1)

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    # num_embeddings += 1
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class CNN_Embedding(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_sizes, c_dropout, k=1):
        super(CNN_Embedding, self).__init__()

        # Embedding
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)

        # CNN
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        
        self.dropout = nn.Dropout(c_dropout)

        self.k = k

        self.pool = DynamicKMaxPooling(k, n_filters)

    def forward(self, dialogue):
        # print(dialogue.size())

        batch_size, timesteps, sent_len = dialogue.size()
        
        c_in = dialogue.view(batch_size * timesteps, sent_len)
        
        embedded = self.embedding(c_in)
                
        #embedded = [batch size * timesteps, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size * timesteps, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size * timesteps, n_filters, sent len - filter_sizes[n]]
        
        # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        pooled = [self.pool(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled = [kmax_pooling(conv, 2, self.k).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size * timesteps, n_filters]
        
        c_out = self.dropout(torch.cat(pooled, dim=1))

        #c_out = [batch size * timesteps, n_filters * len(filter_sizes)]
        
        return c_out