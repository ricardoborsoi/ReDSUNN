import torch
import torch.nn as nn
from model.mymodules import Emitter, Transition, Combiner, RnnEncoder
from data_loader.seq_util import seq_collate_fn, pack_padded_seq
from base import BaseModel

from utils import get_patch_from_embedding
from scipy.io import loadmat, savemat




class DeepMarkovModel(BaseModel):

    def __init__(self,
                 num_bands,
                 num_endmembers,
                 K,
                 sigma_psi,
                 path_load_M0,
                 rnn_type,
                 rnn_layers,
                 rnn_bidirection,
                 use_embedding,
                 train_init,
                 train_M0=False,
                 mean_field=False,
                 reverse_rnn_input=True,
                 sample=True):
        super().__init__()
        rnn_dim = num_endmembers * (K+1)
        
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.K = K
        self.sigma_psi = sigma_psi
        self.z_dim = num_endmembers*(1+K)
        self.input_dim = num_bands
        
        # self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.rnn_bidirection = rnn_bidirection
        self.use_embedding = use_embedding
        self.train_init = train_init
        self.mean_field = mean_field
        self.reverse_rnn_input = reverse_rnn_input
        self.sample = sample

        if use_embedding:
            self.embedding = nn.Linear(num_bands, rnn_dim)
            rnn_input_dim = rnn_dim
        else:
            rnn_input_dim = num_bands


        # load reference endmember matrix and define it as a parameter to pass to models
        M0_init = torch.from_numpy(loadmat(path_load_M0)['M0']).type(torch.FloatTensor)
        self.M0 = nn.Parameter(M0_init, requires_grad=train_M0)

        # instantiate components of DMM ------------------------------
        # generative model
        self.emitter = Emitter(num_bands, num_endmembers, K, self.M0)
        self.transition = Transition(num_endmembers, K, sigma_psi)
        
        # inference model        
        self.combiner = Combiner(num_endmembers, K, rnn_dim, self.M0,
                                 mean_field=mean_field)    
        self.encoder = RnnEncoder(rnn_input_dim, rnn_dim,
                                  n_layer=rnn_layers, drop_rate=0.0,
                                  bd=rnn_bidirection, nonlin='relu',
                                  rnn_type=rnn_type,
                                  reverse_input=reverse_rnn_input)

        # initialize hidden states
        self.mu_p_0, self.logvar_p_0 = self.transition.init_z_0(trainable=train_init)
        self.z_q_0 = self.combiner.init_z_q_0(trainable=train_init)


    def reparameterization(self, mu, logvar):
        if not self.sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, x_reversed, x_seq_lengths):
        T_max      = x.size(1)
        batch_size = x.size(0)

        if self.encoder.reverse_input:
            input = x_reversed
        else:
            input = x

        # recover input from the embedding
        input = get_patch_from_embedding(input, 1)
        
        if self.use_embedding:
            input[:,:,:,0,0] = self.embedding(input[:,:,:,0,0])
        
        # encode each pixel in a patch separately
        h_rnn = torch.zeros((batch_size, T_max, self.rnn_dim*(2 if self.rnn_bidirection else 1), 1, 1))
        h_rnn[:,:,:,0,0] = self.encoder(pack_padded_seq(input[:,:,:,0,0], x_seq_lengths), x_seq_lengths)
        
        z_q_0 = self.z_q_0.expand(batch_size, self.z_dim)
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.z_dim)
        logvar_p_0 = self.logvar_p_0.expand(batch_size, 1, self.z_dim)
        z_prev = z_q_0

        # initialize
        x_recon      = torch.zeros([batch_size, T_max, self.num_bands]).to(x.device)
        mu_q_seq     = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        logvar_q_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        mu_p_seq     = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        logvar_p_seq = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        z_q_seq      = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        z_p_seq      = torch.zeros([batch_size, T_max, self.z_dim]).to(x.device)
        for t in range(T_max):
            # q(z_t | z_{t-1}, x_{t:T})
            mu_q, logvar_q = self.combiner(h_rnn[:,t,:,:,:], x[:,t:T_max,:], z_prev, rnn_bidirection=self.rnn_bidirection)

            zt_q = self.reparameterization(mu_q, logvar_q)
            z_prev = zt_q
            
            # p(z_t | z_{t-1})
            mu_p, logvar_p = self.transition(z_prev)
            zt_p = self.reparameterization(mu_p, logvar_p)
            
            # p(y_t|z_t)
            xt_recon, sigma2_noise = self.emitter(zt_q)
            xt_recon = xt_recon.contiguous()
            
            # pack the results in tensors
            mu_q_seq[:, t, :]     = mu_q
            logvar_q_seq[:, t, :] = logvar_q
            z_q_seq[:, t, :]      = zt_q
            mu_p_seq[:, t, :]     = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p_seq[:, t, :]      = zt_p
            x_recon[:, t, :]      = xt_recon

        mu_p_seq = torch.cat([mu_p_0, mu_p_seq[:, :-1, :]], dim=1)
        logvar_p_seq = torch.cat([logvar_p_0, logvar_p_seq[:, :-1, :]], dim=1)
        z_p_0 = self.reparameterization(mu_p_0, logvar_p_0)
        z_p_seq = torch.cat([z_p_0, z_p_seq[:, :-1, :]], dim=1)
        
        return x_recon, sigma2_noise, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq


    def generate(self, batch_size, seq_len):
        mu_p = self.mu_p_0.expand(batch_size, self.z_dim)
        logvar_p = self.logvar_p_0.expand(batch_size, self.z_dim)
        z_p_seq = torch.zeros([batch_size, seq_len, self.z_dim]).to(mu_p.device)
        mu_p_seq = torch.zeros([batch_size, seq_len, self.z_dim]).to(mu_p.device)
        logvar_p_seq = torch.zeros([batch_size, seq_len, self.z_dim]).to(mu_p.device)
        output_seq = torch.zeros([batch_size, seq_len, self.input_dim]).to(mu_p.device)
        for t in range(seq_len):
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p = self.reparameterization(mu_p, logvar_p)
            xt = self.emitter(z_p)
            mu_p, logvar_p = self.transition(z_p)

            output_seq[:, t, :] = xt
            z_p_seq[:, t, :] = z_p
        return output_seq, z_p_seq, mu_p_seq, logvar_p_seq
