import argparse
import warnings
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.mymodel as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import scipy.io as scio
import time
from utils import plot_abundances_t


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_train', module_data)
    try:
        valid_data_loader = config.init_obj('data_loader_valid', module_data)
    except Exception:
        warnings.warn("Validation dataloader not given.")
        valid_data_loader = None
    try:
        test_data_loader = config.init_obj('data_loader_test', module_data)
    except Exception:
        warnings.warn("Test dataloader not given.")
        test_data_loader = None

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    try:
        metrics = [getattr(module_metric, met) for met in config['metrics']]
    # -------------------------------------------------
    # add flexibility to allow no metric in config.json
    except Exception:
        warnings.warn("No metrics are configured.")
        metrics = None
    # -------------------------------------------------

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # -------------------------------------------------
    # add flexibility to allow no lr_scheduler in config.json
    try:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    except Exception:
        warnings.warn("No learning scheduler is configured.")
        lr_scheduler = None
    # -------------------------------------------------

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      overfit_single_batch=config['trainer']['overfit_single_batch'])

    trainer.train()
    
    # get data and pass it through inference net
    Y = data_loader.dataset.data['sequences']
    x_seq_lengths = data_loader.dataset.data['sequence_lengths']
    x_recon, sigma2_noise, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, \
        mu_p_seq, logvar_p_seq = model(Y, [], x_seq_lengths)
    
    mu_q_seq = mu_q_seq.detach().cpu()
    x_recon  = x_recon.detach().cpu()
    
    # convert to abundances 
    A_hat = torch.softmax(mu_q_seq[:,:,0:model.num_endmembers], dim=2)
    A_hat = A_hat.permute(0,2,1).reshape(data_loader.dataset.nr,
                                         data_loader.dataset.nc, 
                                         model.num_endmembers,
                                         A_hat.shape[1])
    
    # get reconstructed data
    Y_hat = x_recon.permute(0,2,1)
    
    # get psis (variability coefficients)
    psis = mu_q_seq[:,:,model.num_endmembers:].permute(0,2,1)
    psis = psis.reshape(data_loader.dataset.nr*data_loader.dataset.nc, 
                        model.K,
                        model.num_endmembers,
                        psis.shape[2])

    # compute endmember matrices
    N = data_loader.dataset.nr*data_loader.dataset.nc
    L = Y_hat.shape[1]
    P = psis.shape[2]
    
    Mn_hat = torch.zeros((L, P, N, psis.shape[3]))
    if model.K > 0:
        for t in range(psis.shape[3]):
            for i in range(N):
                Psi_n = torch.mm(model.emitter.D[:,0:model.K], 0.01*psis[i,:,:,t])
                Mn_hat[:,:,i,t] = model.M0 * (torch.ones((L,P)) + Psi_n) 
    else:
        for t in range(psis.shape[3]):
            for i in range(N):
                Mn_hat[:,:,i,t] = model.M0
    
    # return the mean abundances and endmembers
    return A_hat, Mn_hat, Y_hat, psis
    



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    # args.add_argument('-c', '--config', default='config_Tahoe.json', type=str,
    #                   help='config file path (default: None)')
    args.add_argument('-c', '--config', default='config_synth_ex1.json', type=str,
                      help='config file path (default: None)')
    # args.add_argument('-c', '--config', default='config_synth_ex2.json', type=str,
    #                   help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--identifier', default=None, type=str,
                      help='unique identifier of the experiment (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    t_start = time.time() # start timer
    A_hat, Mn_hat, Y_hat, psis = main(config)
    t_elapsed = time.time() - t_start # measure elapsed time
    
    # plot abundances
    plot_abundances_t(A=A_hat.detach(), thetitle='title', savepath=None)

    # save results data
    scio.savemat('saved/resultsVRNN_ex_' + args.parse_args().config[0:-5] + '.mat', \
                 {'A_hat_VRNN' : A_hat.cpu().detach().numpy(),
                  'Mn_hat_VRNN' : Mn_hat.cpu().detach().numpy(), 
                  'Y_hat_VRNN' : Y_hat.cpu().detach().numpy(), 
                  'psis_VRNN' : psis.cpu().detach().numpy(),
                  'time_VRNN' : t_elapsed})
    

