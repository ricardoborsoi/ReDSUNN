import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch
from matplotlib import pyplot as plt


def plot_abundances_t(A, thetitle='title', savepath=None):
    ''' plots abundance maps over time
    A : abundance tensor with dimension (nr,nc,P,T) '''    
    P = A.shape[2] # number of endmembers
    T = A.shape[3] # number of time instants
    fig, axs = plt.subplots(T, P)
    for t in range(T):
        for i in range(P):
            axs[t,i].imshow(A[:,:,i,t], cmap='jet', vmin=0, vmax=1) #cmap='gray'
            axs[t,i].axis('off')
    if savepath is not None: # save a figure is specified
        plt.savefig(savepath, dpi=300, format='pdf')
    plt.show()
    

def patch_embedder(Y, psize=1):
    ''' returns an imaage where every pixel is the stack of all bands therein
    Y : N M L tensor with L bands and N by M pixels'''
    exc_size = (psize-1)/2
    assert int(exc_size) == exc_size
    exc_size = int(exc_size)
    dims = Y.shape
    
    Y_embedded = torch.zeros((dims[0], dims[1], (psize**2)*dims[2]))
    for i in range(dims[0]):
        for j in range(dims[1]):
            
            # y_embed = torch.zeros(((psize**2)*dims[2],))
            y_embed = torch.zeros((0,))
            for k in range(i-exc_size, i+exc_size+1):
                for l in range(j-exc_size, j+exc_size+1):
                    idx_r = k
                    idx_c = l
                    
                    # handle symmetric boundary conditions
                    if k < 0:
                        idx_r = -k
                    if k > (dims[0]-1):
                        idx_r = (dims[0]-1) - (k - (dims[0]-1))
                        
                    if l < 0:
                        idx_c = -l
                    if l > (dims[1]-1):
                        idx_c = (dims[1]-1) - (l - (dims[1]-1))
                    
                    y_embed = torch.cat((y_embed,Y[idx_r,idx_c,:].squeeze()))
                        
            Y_embedded[i,j,:] = y_embed
    return Y_embedded


def get_patch_from_embedding(y_embed, psize=1):
    ''' y_embed : batch*T*(patch**2 L), where the last mode has
    pixels in a patch stacked together '''
    L = y_embed.shape[2]/(psize**2)
    assert int(L) == L
    L = int(L)
    T = y_embed.shape[1]
    batch_size = y_embed.shape[0]
    
    y_patch = torch.zeros((batch_size,T,L,psize,psize))
    for b in range(batch_size):
        for t in range(T):
            for i in range(psize):
                for j in range(psize):
                    y_patch[b,t,:,i,j] = \
                        y_embed[b,t,(i*psize+j)*L : (i*psize+j+1)*L]
    return y_patch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        self._data = self._data[self._data.counts != 0]
        return dict(self._data.average)

    def write_to_logger(self, key, value=None):
        assert self.writer is not None
        if value is None:
            self.writer.add_scalar(key, self._data.average[key])
        else:
            self.writer.add_scalar(key, value)
