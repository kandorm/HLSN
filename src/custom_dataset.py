import numpy as np
import torch
from torch.utils import data as data_utils
from itertools import chain

# @deprecated
class DAMICDataset(data_utils.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, targets):
        """Reads source and target sequences from txt files."""
        self.src_seqs = data
        self.trg_seqs = targets
        self.num_total_seqs = len(targets)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        return src_seq, trg_seq

    def __len__(self):
        return self.num_total_seqs

# @deprecated
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        sequences = list(sequences)
        lengths = [len(seq) for seq in sequences]
        u_len = len(sequences[0][0])
        for row in sequences:
            diff = max(lengths) - len(row)
            for i in range(diff):
                row.append([0]*u_len)
        return np.array(sequences), np.array(lengths)

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    src_seqs = torch.from_numpy(src_seqs).long()
    trg_seqs = torch.from_numpy(trg_seqs).float()
    src_lengths = torch.from_numpy(src_lengths).long()
    trg_lengths = torch.from_numpy(trg_lengths).long()

    # print(src_seqs.size())

    return src_seqs, src_lengths, trg_seqs, trg_lengths

# def pad(data, max_d, max_u):
#     # dialog_lengths = [len(dialog) for dialog in data]    
#     # print(dialog_lengths)
#     for row in data:
#         diff = max_d - len(row)
#         for i in range(diff):
#             row.append([0] * max_u)
#     return np.array(data)

# @deprecated
def unpad(data, lengths):
    ret = None
    for i, l in enumerate(lengths):
        if ret is None:
            ret = data[i][0:l]
        else:
            ret = np.append(ret, data[i][0:l], axis=0)
    # print(len(ret))
    # print(sum(lengths))
    return ret

def batch_maker(data, mask, targets, targets_P, batch_size, shuffle=True):
    sequences = list(zip(data, mask, targets, targets_P))
    sequences.sort(key=lambda x: len(x[0]), reverse=True)
    # return mini-batched data and targets
    ret = list(chunks(sequences, batch_size))
    if shuffle:
        np.random.shuffle(ret)
    return ret 
def chunks(l, n):
    head = 0
    for i in range(0, len(l)):
        if i == len(l) -1 or len(l[i][0]) != len(l[i+1][0]) or i - head == n - 1:
            src_seqs = np.array(list(list(zip(*l[head:i + 1]))[0]))
            seq_mask = np.array(list(list(zip(*l[head:i + 1]))[1]))
            trg_seqs = np.array(list(list(zip(*l[head:i + 1]))[2]))
            trg_seqs_P = np.array(list(list(zip(*l[head:i + 1]))[3]))
            src_seqs = torch.from_numpy(src_seqs).long()
            seq_mask = torch.from_numpy(seq_mask).long()
            trg_seqs = torch.from_numpy(trg_seqs).float()
            trg_seqs_P = torch.from_numpy(trg_seqs_P).float()
            yield (src_seqs, seq_mask, trg_seqs, trg_seqs_P)
            head = i + 1

def flattern_result(list_of_lists):
    return list(chain.from_iterable(list_of_lists))
