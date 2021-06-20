# Based on https://arxiv.org/pdf/1404.2188.pdf
# Anton Melnikov

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DynamicKMaxPooling(nn.Module):
    def __init__(self, k_top, L):
        super().__init__()
        # "L is the total  number  of  convolutional  layers
        # in  the  network;
        # ktop is the fixed pooling parameter for the
        # topmost  convolutional  layer" 
        self.k_top = k_top
        self.L = L
    
    def forward(self, X, l):
        # l is the current convolutional layer
        # X is the input sequence
        # s is the length of the sequence
        # (for conv layers, the length dimension is last)
        s = X.size()[2]
        k_ll = ((self.L - l) / self.L) * s
        k_l = round(max(self.k_top, np.ceil(k_ll)))
        out = F.adaptive_max_pool1d(X, k_l)
        return out