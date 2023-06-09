import time
import torch
from torch.distributions import Normal
from model.loss import nll_loss, kl_div


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def nll_metric(output, target, sigma2_noise, mask):
    assert output.dim() == target.dim() == 3
    # assert output.size() == target.size()
    assert mask.dim() == 2
    assert mask.size(1) == output.size(1)
    loss = nll_loss(output, target, sigma2_noise)  # (batch_size, time_step, input_dim)
    loss = mask * loss.sum(dim=-1)  # (batch_size, time_step)
    loss = loss.sum(dim=1, keepdim=True)  # (batch_size, 1)
    return loss

def kl_div_metric(output, target, mask):
    mu1, logvar1 = output
    mu2, logvar2 = target
    assert mu1.size() == mu2.size()
    assert logvar1.size() == logvar2.size()
    assert mu1.dim() == logvar1.dim() == 3
    assert mask.dim() == 2
    assert mask.size(1) == mu1.size(1)
    kl = kl_div(mu1, logvar1, mu2, logvar2)
    kl = mask * kl.sum(dim=-1)
    kl = kl.sum(dim=1, keepdim=True)
    return kl


def bound_eval(output, target, mask):
    x_recon, sigma2_noise, mu_q, logvar_q = output
    x, mu_p, logvar_p = target
    # batch_size = x.size(0)
    neg_elbo = nll_metric(x_recon, x, sigma2_noise, mask) + \
        kl_div_metric([mu_q, logvar_q], [mu_p, logvar_p], mask)
    # tsbn_bound_sum = elbo.div(mask.sum(dim=1, keepdim=True)).sum().div(batch_size)
    bound_sum = neg_elbo.sum().div(mask.sum())
    return bound_sum


def importance_sample(batch_idx, model, x, x_reversed, x_seq_lengths, mask, n_sample=500):
    sample_batch_size = 25
    n_batch = n_sample // sample_batch_size
    sample_left = n_sample % sample_batch_size
    if sample_left == 0:
        n_loop = n_batch
    else:
        n_loop = n_batch + 1

    ll_estimate = torch.zeros(n_loop).to(x.device)

    start_time = time.time()
    for i in range(n_loop):
        if i < n_batch:
            n_repeats = sample_batch_size
        else:
            n_repeats = sample_left

        x_tile = x.repeat_interleave(repeats=n_repeats, dim=0)
        x_reversed_tile = x_reversed.repeat_interleave(repeats=n_repeats, dim=0)
        x_seq_lengths_tile = x_seq_lengths.repeat_interleave(repeats=n_repeats, dim=0)
        mask_tile = mask.repeat_interleave(repeats=n_repeats, dim=0)

        x_recon, sigma2_noise, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
            model(x_tile, x_reversed_tile, x_seq_lengths_tile)

        q_dist = Normal(mu_q_seq, logvar_q_seq.exp().sqrt())
        p_dist = Normal(mu_p_seq, logvar_p_seq.exp().sqrt())
        log_qz = q_dist.log_prob(z_q_seq).sum(dim=-1) * mask_tile
        log_pz = p_dist.log_prob(z_q_seq).sum(dim=-1) * mask_tile
        log_px_z = -1 * nll_loss(x_recon, x_tile, sigma2_noise).sum(dim=-1) * mask_tile
        ll_estimate_ = log_px_z.sum(dim=1, keepdim=True) + \
            log_pz.sum(dim=1, keepdim=True) - \
            log_qz.sum(dim=1, keepdim=True)

        ll_estimate[i] = ll_estimate_.sum().div(mask.sum())

    ll_estimate = ll_estimate.sum().div(n_sample)
    print("%s-th batch, importance sampling took %.4f seconds." % (batch_idx, time.time() - start_time))

    return ll_estimate
