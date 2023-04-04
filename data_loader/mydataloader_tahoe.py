#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat, savemat
import mat73

import torch
from torch.utils.data import Dataset
from utils import patch_embedder


# ingest training/validation/test data from disk
def load_data(data_dir):
    
    # load dataset
    mat_contents1 = mat73.loadmat(data_dir)
    
    # order as L*N by T, one subsequence per pixel? how to deal with psis correlated?
    # one subsequence per image region
    
    Y = torch.from_numpy(np.stack(mat_contents1['Y'], axis=0)).permute(2,0,1)
    N, T = Y.shape[0], Y.shape[1]
    nr, nc = 110, 150
    
    # embed image patches for spatial-spectral processing
    Yim_embedded = torch.zeros((6,173,nr,nc))
    for t in range(6):
        Yim_t = torch.from_numpy(Y[:,t,:].T.numpy().reshape((173,150,110), order='F'))
        Yim_t = Yim_t.permute(0,2,1)
        Yim_embedded[t,:,:,:] = patch_embedder(Yim_t.permute(1,2,0), 1).unsqueeze(0).permute(0,3,1,2)
    
    Yim_embedded = Yim_embedded.reshape((6,173,N)).permute(2,0,1)
    
    seq_lens = T*torch.ones((N,), dtype=int) # length of each sequence
    
    XX = {"sequences" : Yim_embedded.type(torch.Tensor), \
          "sequence_lengths" : seq_lens.to(device=torch.Tensor().device), \
          "true_abundances" : None, \
          "true_endmembers" : None, \
          "true_M0" : None }
    dset = {"train":XX, "valid":XX, "test":XX}
    
    return dset



class TahoeDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split

        self.data = load_data(data_dir)[split]
        self.seq_lengths = self.data['sequence_lengths']
        self.seq = self.data['sequences']
        self.n_seq = len(self.seq_lengths)
        self.n_time_slices = float(torch.sum(self.seq_lengths))
        
        self.nr = 110
        self.nc = 150
        

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        return idx, self.seq[idx], self.seq_lengths[idx]
