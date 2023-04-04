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
    mat_contents1 = loadmat(data_dir)
    
    # order as L*N by T, one subsequence per pixel? how to deal with psis correlated?
    # one subsequence per image region
    
    Y = torch.from_numpy(mat_contents1['Y'])
    T = Y.shape[3]
    nr, nc = Y.shape[1], Y.shape[2]
    L = Y.shape[0]
    N = nr*nc
    
    # embed image patches for spatial-spectral processing
    Yim_embedded = torch.zeros((T,L,nr,nc))
    for t in range(T):
        Yim_t = Y[:,:,:,t]
        Yim_embedded[t,:,:,:] = patch_embedder(Yim_t.permute(1,2,0), 1).unsqueeze(0).permute(0,3,1,2)
    
    Yim_embedded = Yim_embedded.reshape((T,L,N)).permute(2,0,1)
    
    seq_lens = T*torch.ones((N,), dtype=int) # length of each sequence
    
    XX = {"sequences" : Yim_embedded.type(torch.Tensor), \
          "sequence_lengths" : seq_lens.to(device=torch.Tensor().device), \
          "true_abundances" : mat_contents1['A'], \
          "true_endmembers" : mat_contents1['M_nt'], \
          "true_M0" : mat_contents1['M'] }
    dset = {"train":XX, "valid":XX, "test":XX}

    return dset




class SynthEx2Dataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split

        self.data = load_data(data_dir)[split]
        self.seq_lengths = self.data['sequence_lengths']
        self.seq = self.data['sequences']
        self.n_seq = len(self.seq_lengths)
        self.n_time_slices = float(torch.sum(self.seq_lengths))
        
        self.nr = 50
        self.nc = 50
        
    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        return idx, self.seq[idx], self.seq_lengths[idx]



        
