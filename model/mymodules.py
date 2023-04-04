#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from data_loader.seq_util import pad_and_reverse
import math

import numpy as np
from scipy.fft import dct
from utils import get_patch_from_embedding


"""
Generative modules
"""


class Emitter(nn.Module):
    """
    Parameterize the Gaussian observation likelihood `p(x_t | z_t)`, where
    variable x_t are pixels and z_t=[a_t,psi_t] the abundances and variability 
    scalings.

    Parameters
    ----------
    num_bands: int
        number of spectral bands
    num_endmembers: int
        number of endmembers
    K: int
        rank of the DFT basis used to represent the variability
    M0: learnable parameter
        initialization of the reference EM matrix

    Returns
    -------
        A valid probability that parameterizes the
        Gaussian distribution `p(x_t | z_t)`
    """
    def __init__(self, num_bands, num_endmembers, K, M0):
        super().__init__()
        self.z_dim = num_endmembers*(1+K)
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        
        self.M0 = M0 #nn.Parameter(M0_init, requires_grad=True)
        
        self.P = self.M0.shape[1]
        self.L = self.M0.shape[0]
        self.K = K # rank of Psi
        
        self.D = torch.from_numpy(dct(np.eye(num_bands), axis=0)).type(torch.FloatTensor).T
        self.log_sigma2_noise = nn.Parameter(torch.log(torch.tensor([0.0001])), \
                                         requires_grad=True) # measurement noise variance 
    

    def forward(self, z_t):
        # z_t is batch by P now
        batch_size = z_t.shape[0]
        P, K, L = self.P, self.K, self.L
        N = 1 # pixelwise generative movel
        
        # get abundances
        z_t_abund = z_t[:,0:P*N].reshape((batch_size,N,P))
        a_t = torch.softmax(z_t_abund, dim=2)
        
        # get psis
        z_t_psis = z_t[:,P*N:P*N+N*P*K].reshape((batch_size,N,K,P))
        Y = torch.zeros((batch_size, L*N))
        for i in range(batch_size):
            for j in range(N):
                Psi_n = torch.mm(self.D[:,0:K], z_t_psis[i,j,:,:])
                Mn = self.M0 * (torch.ones((L,P)) + 0.01*Psi_n) # the 0.01 scales down the variability co compensate the higher amplitudes in D
                # generate the pixel
                Y[i, j*L:(j+1)*L] = torch.mv(Mn, a_t[i,j,:]).unsqueeze(0)
        
        # return mean image and noise variance
        return Y, torch.exp(self.log_sigma2_noise)
    
    
    
    

class Transition(nn.Module):
    """
    Parameterize the diagonal Gaussian latent transition probability
    `p(z_t | z_{t-1})`, where variable z_t=[a_t,psi_t] the abundances and variability 
    scalings. (maybe include change detection too)
    
    Parameters
    ----------
    num_endmembers: int
        number of endmembers
    K: int
        rank of the DCT approximation
    sigma_psi: float
        variance of innovation noise for the scaling factors
    train_sigma_psi: bool
        whether sigma_psi is trainable

    Returns
    -------
    mu: tensor (b, z_dim)
        Mean that parameterizes the Gaussian
    logvar: tensor (b, z_dim)
        Log-variance that parameterizes the Gaussian
    """
    def __init__(self, num_endmembers, K, sigma_psi, train_sigma_psi=False):
        super().__init__()
        self.num_endmembers = num_endmembers
        self.K = K
        self.z_dim = num_endmembers*(K+1)
        
        if train_sigma_psi is True:
            self.sigma_psi = nn.Parameter(torch.tensor([sigma_psi]), requires_grad=True)
        else:
            self.sigma_psi = torch.tensor([sigma_psi])
        
        # compute the logvar
        self.lin_v = nn.Sequential(nn.Linear(num_endmembers, num_endmembers),
                                   nn.ReLU(),
                                   nn.Linear(num_endmembers, 1),)
        
    def init_z_0(self, trainable=True): # this returns mean and logvar of p(z_0)
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable), \
            nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)

    def forward(self, z_t_1, u_t_1=None):
        batch_size = z_t_1.shape[0]
        
        mu_a   = z_t_1[:,0:self.num_endmembers]
        mu_psi = z_t_1[:,self.num_endmembers:self.num_endmembers*(self.K+1)]
        mu     = torch.cat((mu_a,mu_psi),dim=1) 
        
        logvar_psi = torch.log(self.sigma_psi*torch.ones((batch_size,self.K*self.num_endmembers)))
        logvar_a   = torch.log(0.1*torch.exp(self.lin_v(mu_a))*torch.ones((1,self.num_endmembers)))
        logvar     = torch.cat((logvar_a,logvar_psi), dim=1)
        
        return mu, logvar


"""
Inference modules
"""


class Combiner(nn.Module):
    """
    Parameterize variational distribution `q(z_t | z_{t-1}, x_{t:T})`
    a diagonal Gaussian distribution

    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    rnn_dim: int
        Dim. of RNN hidden states

    Returns
    -------
    mu: tensor (b, z_dim)
        Mean that parameterizes the variational Gaussian distribution
    logvar: tensor (b, z_dim)
        Log-var that parameterizes the variational Gaussian distribution
    """
    def __init__(self, num_endmembers, K, rnn_dim, M0, mean_field=False):
        super().__init__()
        self.num_endmembers = num_endmembers
        self.K = K
        self.z_dim = num_endmembers*(K+1)
        self.M0 = M0
        self.rnn_dim = rnn_dim
        self.mean_field = mean_field
        self.D = torch.from_numpy(dct(np.eye(M0.shape[0]), axis=0)).type(torch.FloatTensor)

        # initialize learnable parameters
        self.conv1 = nn.Conv2d(rnn_dim, num_endmembers, kernel_size=1)
        self.conv3 = nn.Conv2d(num_endmembers, num_endmembers, kernel_size=1)
        self.conv4 = nn.Conv2d(rnn_dim, num_endmembers, kernel_size=1)

        self.lin11 = nn.Linear(rnn_dim, num_endmembers)
        self.lin22 = nn.Linear(rnn_dim, num_endmembers*K)
        
        # check if there is variability
        if K > 0:
            self.conv2 = nn.Conv2d(rnn_dim, num_endmembers*K, kernel_size=1)
            self.conv5 = nn.Conv2d(rnn_dim, num_endmembers*K, kernel_size=1)
        else:
            self.conv2 = lambda z: torch.zeros((z.shape[0],0,1,1))
            self.conv5 = lambda z: torch.zeros((z.shape[0],0,1,1))
        
        self.w_a   = nn.Parameter(torch.Tensor([1., 1., 0., 0.]), requires_grad=True)
        self.w_psi = nn.Parameter(torch.Tensor([1., 0., 0.]), requires_grad=True)
        

    def init_z_q_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)
        

    def forward(self, h_rnn, x_t_T, z_t_1=None, rnn_bidirection=False):
        """
        - z_t_1 : tensor (b, z_dim), containing [a_t_1,psi_t_1]
        - h_rnn : tensor of size (b, rnn_dim, 1, 1), 
                  containing the RNN activations on patches centered on b at time t
        - x_t_T : tensor (b, T-t+1, L) with embeddings of patches
                  of pixels
        """
        batch_size = x_t_T.shape[0]
        P = self.num_endmembers
        L = self.M0.shape[0]
        K = self.K

        # first recover the endmember matrices
        Mn = torch.zeros((batch_size,L,P))
        if z_t_1 is not None:
            z_t_psis = z_t_1[:,P:P+P*K].reshape((batch_size,K,P))
            for i in range(batch_size):
                Psi_n = torch.mm(self.D[:,0:K], z_t_psis[i,:,:])
                Mn[i,:,:] = self.M0 * (torch.ones((L,P)) + 0.01*Psi_n) 
        else:
            for i in range(batch_size):
                Mn[i,:,:] = self.M0
        
        # use the predicted EMs to preliminarily unmix x_t
        x_t_T_unembedded = get_patch_from_embedding(x_t_T, 1)
        a_t_lin = torch.bmm(torch.linalg.pinv(Mn), \
                            x_t_T_unembedded[:,0,:,0,0].unsqueeze(2)).squeeze(2)
        
        # get the predicted abundances a_t_1
        a_t_1   = torch.softmax(z_t_1[:,0:P], dim=1)
        psi_t_1 = z_t_1[:,P:P+P*K]
        
        # estimate changes from the preliminary abundances and a_t_1
        tmp_a_t = torch.nn.functional.relu(a_t_lin, inplace=False)
        tmp_a_t = tmp_a_t / (tmp_a_t.sum(-1, keepdim=True) + 1e-4)
        u_t = (1/2)*torch.sum(torch.abs((a_t_1-tmp_a_t)), dim=1)
        u_t = torch.mm(u_t.unsqueeze(1), torch.ones(1,P))
        
        # get info from RNN for a and psi
        if rnn_bidirection:
            h_rnn_sp_a   = torch.mean(0.5*self.conv1(h_rnn[:,:self.rnn_dim,:,:]) + \
                                      0.5*self.conv1(h_rnn[:,self.rnn_dim:,:,:]), dim=(2,3))
            h_rnn_sp_psi = torch.mean(0.5*self.conv2(h_rnn[:,:self.rnn_dim,:,:]) + \
                                      0.5*self.conv2(h_rnn[:,self.rnn_dim:,:,:]), dim=(2,3))
            h_rnn_px_a   = 0.5*self.lin11(h_rnn[:,:self.rnn_dim,0,0]) + \
                           0.5*self.lin11(h_rnn[:,self.rnn_dim:,0,0])   
            h_rnn_px_psi = 0.5*self.lin22(h_rnn[:,:self.rnn_dim,0,0]) + \
                           0.5*self.lin22(h_rnn[:,self.rnn_dim:,0,0])
        else:
            h_rnn_sp_a   = torch.mean(self.conv1(h_rnn), dim=(2,3))
            h_rnn_sp_psi = torch.mean(self.conv2(h_rnn), dim=(2,3))
            h_rnn_px_a   = self.lin11(h_rnn[:,:,0,0])
            h_rnn_px_psi = self.lin22(h_rnn[:,:,0,0])
        
        # now combine the info fromlin linear, nonlinear, spatial, and t_1
        a_t = (self.w_a[0] * a_t_1) * (1-u_t) \
            + (u_t) * (self.w_a[1] * a_t_lin + \
                       self.w_a[2] * h_rnn_px_a + \
                       self.w_a[3] * h_rnn_sp_a)
        
        # map from Dirichlet to softmax basis
        a_t = torch.nn.functional.relu(a_t)
        mu_a = torch.log(a_t + 1e-8)
        mu_a = mu_a - (1/P) * mu_a.sum(1, keepdim=True)
        
        # also do the same for psi, but without change detection
        mu_psi = self.w_psi[0] * psi_t_1 + \
                 self.w_psi[1] * h_rnn_px_psi + \
                 self.w_psi[2] * h_rnn_sp_psi 
        
        # now compute the log-variances
        if rnn_bidirection:
            logvar_a   = 0.5*self.conv4(h_rnn[:,:self.rnn_dim,:,:]) + \
                         0.5*self.conv4(h_rnn[:,self.rnn_dim:,:,:])
            logvar_psi = 0.5*self.conv5(h_rnn[:,:self.rnn_dim,:,:]) + \
                         0.5*self.conv5(h_rnn[:,self.rnn_dim:,:,:])
            logvar_a   = torch.mean(logvar_a, dim=(2,3))
            logvar_psi = torch.mean(logvar_psi, dim=(2,3))
        else:
            logvar_a   = self.conv4(h_rnn)
            logvar_psi = self.conv5(h_rnn)
            logvar_a   = torch.mean(logvar_a, dim=(2,3))
            logvar_psi = torch.mean(logvar_psi, dim=(2,3))
            
        # concatenate results
        mu = torch.cat((mu_a,mu_psi),dim=1)
        logvar = 0.1 * torch.cat((logvar_a,logvar_psi),dim=1)
        
        return mu, logvar




class RnnEncoder(nn.Module):
    """
    RNN encoder that outputs hidden states h_t using x_{t:T}

    Parameters
    ----------
    input_dim: int
        Dim. of inputs
    rnn_dim: int
        Dim. of RNN hidden states
    n_layer: int
        Number of layers of RNN
    drop_rate: float [0.0, 1.0]
        RNN dropout rate between layers
    bd: bool
        Use bi-directional RNN or not

    Returns
    -------
    h_rnn: tensor (b, T_max, rnn_dim * n_direction)
        RNN hidden states at every time-step
    """
    def __init__(self, input_dim, rnn_dim, n_layer=1, drop_rate=0.0, bd=False,
                 nonlin='relu', rnn_type='rnn', orthogonal_init=False,
                 reverse_input=True):
        super().__init__()
        self.n_direction = 1 if not bd else 2
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.drop_rate = drop_rate
        self.bd = bd
        self.nonlin = nonlin
        self.reverse_input = reverse_input

        if not isinstance(rnn_type, str):
            raise ValueError("`rnn_type` should be type str.")
        self.rnn_type = rnn_type
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim,
                              nonlinearity=nonlin, batch_first=True,
                              bidirectional=bd, num_layers=n_layer,
                              dropout=drop_rate)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_dim,
                              batch_first=True,
                              bidirectional=bd, num_layers=n_layer,
                              dropout=drop_rate)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=rnn_dim,
                               batch_first=True,
                               bidirectional=bd, num_layers=n_layer,
                               dropout=drop_rate)
        else:
            raise ValueError("`rnn_type` must instead be ['rnn', 'gru', 'lstm'] %s"
                             % rnn_type)

        if orthogonal_init:
            self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def calculate_effect_dim(self):
        return self.rnn_dim * self.n_direction

    def init_hidden(self, trainable=True):
        if self.rnn_type == 'lstm':
            h0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim), requires_grad=trainable)
            c0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim), requires_grad=trainable)
            return h0, c0
        else:
            h0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim), requires_grad=trainable)
            return h0

    def forward(self, x, seq_lengths):
        """
        x: pytorch packed object
            input packed data; this can be obtained from
            `util.get_mini_batch()`
        h0: tensor (n_layer * n_direction, b, rnn_dim)
        seq_lengths: tensor (b, )
        """
        _h_rnn, _ = self.rnn(x)
        if self.reverse_input:
            h_rnn = pad_and_reverse(_h_rnn, seq_lengths)
        else:
            h_rnn, _ = nn.utils.rnn.pad_packed_sequence(_h_rnn, batch_first=True)
        return h_rnn













