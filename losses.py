"""Summarize all loss of neural network."""
import math
import numpy as np
from torch.nn import functional as F
import torch.nn
import torch
from torch.distributions import Normal
import neural_network
from lie_tools import logsumexp


def final_loss(bce_loss, mu, logvar):
    """This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def bce_on_intensities(x, recon_x, scale_b):
    """BCE summed over the voxels intensities.
    scale_b: plays role of loss' weighting factor.
    """
    bce = torch.sum(
        F.binary_cross_entropy(recon_x, x) / scale_b.exp() + 2 * scale_b)
    return bce


def mse_on_intensities(x, recon_x, scale_b):
    """MSE summed over the voxels intensities.
    scale_b: plays role of loss' weighting factor.
    """
    print(min(recon_x))
    print(max(recon_x))
    print(min(x))
    print(max(x))
    mse = F.mse_loss(recon_x, x, reduction='sum') / scale_b
    return mse


def mse_on_features(feature, recon_feature, logvar):
    """MSE over features of FC layer of Discriminator.
    sigma2: plays role of loss' weighting factor.
    """
    mse = F.mse_loss(recon_feature, feature) / (2 * logvar.exp())
    mse = torch.mean(mse)
    return mse


def kullback_leibler(mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def kullback_leibler_circle(mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    mu_circle = mu[:, :2]
    logvar_circle = logvar[:, :2]
    mu_else = mu[:, 2:]
    logvar_else = logvar[:, 2:]

    kld_circle_attractive = -0.5 * torch.sum(
        1 + logvar_circle - mu_circle.pow(2) - logvar_circle.exp())
    kld_circle_repulsive = 0.5 * torch.sum(
        1 + logvar_circle - mu_circle.pow(2) - logvar_circle.exp())

    kld_else = -0.5 * torch.sum(
        1 + (logvar_else - (-0.6))
        - mu_else.pow(2) / (0.5**2) - logvar_else.exp() / 0.5**2)

    kld = kld_circle_attractive + kld_circle_repulsive + kld_else
    return kld


def on_circle(mu, logvar):
    mu_circle = mu[:, :2]

    on_circle = 1000 * (torch.sum(mu_circle**2, dim=1) - 1) ** 2
    on_circle = torch.sum(on_circle)
    on_circle = on_circle - 0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp())
    return on_circle


def vae_loss(x, recon_x, scale_b, mu, logvar):
    bce = bce_on_intensities(x, recon_x, scale_b)
    kld = kullback_leibler(mu, logvar)
    return bce + kld


def construct_v(mu):
    v = mu.double().view(-1, 2, 3)
    v1, v2 = v[:, 0], v[:, 1]
    return s2s2_gram_schmidt(v1, v2).float()


def so3_kullbackleibler(mu, logvar):
    v = construct_v(mu)
    var_mat = neural_network.rodrigues(logvar)
    rot_mats_no_var, _, logvar_rot_eul = neural_network.reparametrize_so3(
        False, mu, logvar)
    rot_mats = neural_network.add_logvar(rot_mats_no_var, logvar_rot_eul)
    log_q_z_x = log_posterior(v, rot_mats, logvar, k=3)
    log_p_z = log_prior(var_mat)
    kl = log_q_z_x - log_p_z
    return kl.mean(0)


def log_posterior(v, rot_mats, logvar, k):
    theta = v.norm(p=2, dim=-1, keepdim=True)  # [n,B,1]
    u = v / theta  # [n,B,3]

    angles = 2 * math.pi * torch.arange(
        -k, k+1, device=u.device, dtype=v.dtype)  # [2k+1]

    theta_hat = theta[..., None, :] + angles[:, None]  # [n,B,2k+1,1]

    clamp = 1e-3
    x = u[..., None, :] * theta_hat  # [n,B,2k+1,3]

    # [n,(2k+1),B,3] or [n,(2k+1),B]

    log_p = reparametrize(logvar, rot_mats).contiguous()

    if len(log_p.size()) == 4:
        log_p = log_p.sum(-1)  # [n,(2k+1),B]

    log_p = log_p.permute([0, 2, 1])  # [n,B,(2k+1)]

    theta_hat_squared = torch.clamp(theta_hat ** 2, min=clamp)

    log_p.contiguous()
    cos_theta_hat = torch.cos(theta_hat)

    # [n,B,(2k+1),1]
    log_vol = torch.log(theta_hat_squared /
                        torch.clamp(2 - 2 * cos_theta_hat, min=clamp))
    log_p = log_p + log_vol.sum(-1)
    log_p = logsumexp(log_p, -1)

    return log_p


def log_prior(var_mat):
    prior = torch.tensor([- np.log(8 * (np.pi ** 2))],
                         device=var_mat.device)
    return prior.expand_as(var_mat[..., 0, 0])


def reparametrize(logvar, nsample_z):
    return Normal(torch.zeros_like(logvar), logvar).log_prob(nsample_z).sum(-1)


def s2s2_gram_schmidt(v1, v2):
    """Normalise 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix."""
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)
