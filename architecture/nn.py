"""This file is creating Convolutional Neural Networks."""

import functools
import os

import cryo_dataset as ds
import numpy as np
import torch
import torch.nn as nn
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from pinchon_hoggan_dense import Jd, rot_mat

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

CUDA = torch.cuda.is_available()


def test():
    if CUDA:
        path_vae = "Cryo/VAE_Cryo_V3/vae_parameters.json"
    else:
        path_vae = "vae_parameters.json"
    PATHS, SHAPES, CONSTANTS, SEARCH_SPACE, _ = ds.load_parameters(path_vae)
    CONSTANTS.update(SEARCH_SPACE)
    CONSTANTS["latent_space_definition"] = 1
    CONSTANTS["latent_dim"] = 10
    enc = EncoderConv(CONSTANTS)
    A = torch.zeros(200, 1, 64, 64, 64)
    B = enc.forward(A)
    print(B[2].shape)
    dec = DecoderConv(CONSTANTS)
    C = dec.forward(B[2])
    return C


DEVICE = torch.device("cuda" if CUDA else "cpu")

NN_CONV = {2: nn.Conv2d, 3: nn.Conv3d}
NN_CONV_TRANSPOSE = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}

NN_BATCH_NORM = {2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

LATENT_SPACE_DEFINITION = {0: "R^L", 1: "SO(3)xR^L", 2: "SO(3)"}

OUT_PAD = 0


def conv_parameters(conv_dim, kernel_size, stride, padding, dilation):
    """
    Construction of arrays of constants for 2d or 3d problems.

    Parameters
    ----------
    conv_dim : int, 2 or 3 for 2d or 3d problems.
    kernel_size : int, kernel size.
    stride : int, stride.
    padding : int, padding.
    dilation : int, dilation.


    Returns
    -------
    kernel_size : array, kernel size for a 2d or 3d problem
    stride : array, stride for a 2d or 3d problem
    padding : array, padding for a 2d or 3d problem
    """
    if type(kernel_size) is int:
        kernel_size = np.repeat(kernel_size, conv_dim)
    if type(stride) is int:
        stride = np.repeat(stride, conv_dim)
    if type(padding) is int:
        padding = np.repeat(padding, conv_dim)
    if type(dilation) is int:
        dilation = np.repeat(dilation, conv_dim)
    if len(kernel_size) != conv_dim:
        raise ValueError

    if len(stride) != conv_dim:
        raise ValueError
    if len(padding) != conv_dim:
        raise ValueError
    if len(dilation) != conv_dim:
        raise ValueError
    return kernel_size, stride, padding, dilation


def conv_transpose_input_size(
    out_shape,
    in_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    output_padding=OUT_PAD,
):
    """
    Compute the in_shape of a layer by knowing the output shape.

    Parameters
    ----------
    out_shape : tuple, out shape of the layer.
    in_channels : int, number of in channel.
    kernel_size : int, kernel size.
    stride : int,  stride.
    padding : int,padding.
    dilation : int, dilation.
    output_padding : int optional, out pad, the default is OUT_PAD.

    Returns
    -------
    tuple, shape of the information before passing the layer.
    """
    conv_dim = len(out_shape[1:])
    kernel_size, stride, padding, dilation = conv_parameters(
        conv_dim, kernel_size, stride, padding, dilation
    )
    if type(output_padding) is int:
        output_padding = np.repeat(output_padding, conv_dim)

    def one_dim(i_dim):
        """Inverts the formula giving the output shape."""
        shape_i_dim = (
            (
                out_shape[i_dim + 1]
                + 2 * padding[i_dim]
                - dilation[i_dim] * (kernel_size[i_dim] - 1)
                - output_padding[i_dim]
                - 1
            )
            // stride[i_dim]
        ) + 1

        if shape_i_dim % 1 != 0:
            raise ValueError
        return int(shape_i_dim)

    in_shape = [one_dim(i_dim) for i_dim in range(conv_dim)]
    in_shape = tuple(in_shape)

    return (in_channels,) + in_shape


def conv_output_size(in_shape, out_channels, kernel_size, stride, padding, dilation):
    """
    Compute the output shape by knowing the input shape of a layer

    Parameters
    ----------
    in_shape : tuple, shape of the input of the layer.
    out_channels : int, number of output channels.
    kernel_size : int, kernel size.
    stride : int,  stride.
    padding : int,padding.
    dilation : int, dilation.
    Returns
    -------
    out_shape : tuple, shape of the output of the layer.
    """
    out_shape = conv_transpose_input_size(
        out_shape=in_shape,
        in_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=0,
        dilation=dilation,
    )
    # out_shape = (out_shape[0], out_shape[1], out_shape[2])
    return out_shape


class EncoderConv(nn.Module):
    """This class compute the Encoder"""

    def __init__(self, config):
        """
        Initialization of the encoder.

        Parameters
        ----------
        config : dic, principal constants to build a encoder.

        Returns
        -------
        None.

        """
        super(EncoderConv, self).__init__()
        self.config = config
        self.latent_dim = int(config["latent_dim"] / 2)
        self.img_shape = config["img_shape"]
        self.conv_dim = config["conv_dim"]
        self.n_blocks = config["n_enc_lay"]
        self.enc_c = config["enc_c"]
        self.enc_ks = config["enc_ks"]
        self.enc_str = config["enc_str"]
        self.enc_pad = config["enc_pad"]
        self.enc_dil = config["enc_dil"]
        self.nn_conv = NN_CONV[self.conv_dim]
        self.nn_batch_norm = NN_BATCH_NORM[self.conv_dim]
        self.latent_space_definition = config["latent_space_definition"]
        self.z0 = torch.zeros(self.latent_dim).to(DEVICE)
        self.z0[0] = 1
        self.z1 = torch.zeros(self.latent_dim).to(DEVICE)
        self.z1[1] = 1

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # encoder
        self.blocks = torch.nn.ModuleList()

        next_in_channels = self.img_shape[0]
        next_in_shape = self.img_shape
        for i in range(self.n_blocks):
            enc_c_factor = 2 ** i
            enc = self.nn_conv(
                in_channels=next_in_channels,
                out_channels=self.enc_c * enc_c_factor,
                kernel_size=self.enc_ks,
                stride=self.enc_str,
                padding=self.enc_pad,
            )
            bn = self.nn_batch_norm(enc.out_channels)

            self.blocks.append(enc)
            self.blocks.append(bn)
            enc_out_shape = self.enc_conv_output_size(
                in_shape=next_in_shape, out_channels=enc.out_channels
            )
            next_in_shape = enc_out_shape
            next_in_channels = enc.out_channels

        self.last_out_shape = next_in_shape

        self.fcs_infeatures = functools.reduce(
            (lambda x, y: x * y), self.last_out_shape
        )

        self.fc1 = nn.Linear(
            in_features=self.fcs_infeatures, out_features=self.latent_dim
        )

        self.fc2 = nn.Linear(
            in_features=self.fcs_infeatures, out_features=self.latent_dim
        )

        self.fc_nn_so3_mu = nn.Linear(in_features=self.fcs_infeatures, out_features=3)

        self.fc_nn_so3_logvar = nn.Linear(
            in_features=self.fcs_infeatures, out_features=3
        )

        self.fc_so3_latent_space = nn.Linear(
            in_features=3, out_features=self.latent_dim
        )

    def enc_conv_output_size(self, in_shape, out_channels):
        """
        Compute the output shape of a layer

        Parameters
        ----------
        in_shape : tuple, input shape
        out_channels : int, number of channels.

        Returns
        -------
        out_shape : tuple, shape of the output of the layer

        """
        return conv_output_size(
            in_shape,
            out_channels,
            kernel_size=self.enc_ks,
            stride=self.enc_str,
            padding=self.enc_pad,
            dilation=self.enc_dil,
        )

    def forward(self, x):
        """
        Compute the passage through the neural network

        Parameters
        ----------
        x : tensor, image or voxel.

        Returns
        -------
        mu : tensor, latent space mu.
        logvar : tensor, latent space sigma.

        """
        h = x
        for i in range(self.n_blocks):
            h = self.blocks[2 * i](h)
            h = self.blocks[2 * i + 1](h)
            h = self.leakyrelu(h)
        h = h.view(-1, self.fcs_infeatures)
        if self.latent_space_definition == 0:
            mu = self.fc1(h)
            logvar = self.fc2(h)
            z = reparametrize(mu, logvar)
            return mu, logvar, z, torch.zeros(mu.shape)
        mu = self.fc_nn_so3_mu(h)
        logvar = self.fc_nn_so3_logvar(h)
        if self.latent_space_definition == 1:
            matrix = transform_into_so3(mu, logvar, self.latent_dim)
            z0 = torch.matmul(matrix, self.z0)
            z1 = torch.matmul(matrix, self.z1)
            z = torch.cat((z0, z1), dim=1)
            return mu, logvar, z, matrix
        rotvec = reparametrize(mu, logvar)
        return mu, logvar, rotvec, torch.zeros(mu.shape)


SO3 = SpecialOrthogonal(3, point_type="vector")


def transform_into_so3(mu, logvar, latent_dim, n_samples=1):
    n_batch_data, latent_shape = mu.shape
    sigma = logvar.mul(0.5).exp_()
    if CUDA:
        eps = torch.cuda.FloatTensor(n_samples, n_batch_data, latent_shape).normal_()
    else:
        eps = torch.FloatTensor(n_samples, n_batch_data, latent_shape).normal_()
    eps = eps.reshape(eps.shape[1:])
    tang_mu = eps * sigma
    z = mu + tang_mu
    matrices = rot_mat_tensor(z, latent_dim)
    return matrices


def rot_mat_tensor(tens, latent_dim):
    batch_size = tens.shape[0]
    J = Jd[int((latent_dim - 1) / 2)].to(DEVICE)
    matrices = torch.Tensor(batch_size, latent_dim, latent_dim).to(DEVICE)
    for i in range(batch_size):
        alpha = tens[i][0]
        beta = tens[i][1]
        gamma = tens[i][2]
        matrices[i] = rot_mat(alpha, beta, gamma, int((latent_dim - 1) / 2), J)
    return matrices


class DecoderConv(nn.Module):
    """This class compute the decoder"""

    def dec_conv_transpose_input_size(self, out_shape, in_channels):
        """
        Compute the in_shape of a layer by knowing the output shape.

        Parameters
        ----------
        out_shape : tuple, out shape of the layer.
        in_channels : int, number of in channel.

        Returns
        -------
        tuple, shape of the information before passing the layer.

        """
        return conv_transpose_input_size(
            out_shape=out_shape,
            in_channels=in_channels,
            kernel_size=self.dec_ks,
            stride=self.dec_str,
            padding=self.dec_pad,
            dilation=self.dec_dil,
        )

    def block(self, out_shape, dec_c_factor):
        """
        Compute every layer

        Parameters
        ----------
        out_shape : tuple, shape of the output of the layer.
        dec_c_factor : int, decode factor.

        Returns
        -------
        batch_norm : layer, layer of the NN.
        conv_transpose : layer, layer of the NN.
        in_shape : tuple, shape of the input of the layer

        """
        out_channels = out_shape[0]
        in_channels = self.dec_c * dec_c_factor

        batch_norm = self.nn_batch_norm(num_features=out_channels, eps=1.0e-3)

        conv_transpose = self.nn_conv_transpose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.dec_ks,
            stride=self.dec_str,
        )

        in_shape = self.dec_conv_transpose_input_size(
            out_shape=out_shape, in_channels=in_channels
        )
        return batch_norm, conv_transpose, in_shape

    def end_block(self, out_shape, dec_c_factor):
        """
        Compute the last layer of the NN

        Parameters
        ----------
        out_shape : tuple, out shape
        dec_c_factor : int, decode factor

        Returns
        -------
        conv_transpose : torch.nn.modules.conv.ConvTranspose, a layer of my NN
        in_shape : tuple, input shape of the layer.

        """
        out_channels = out_shape[0]
        in_channels = self.dec_c * dec_c_factor

        conv_transpose = self.nn_conv_transpose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.dec_ks,
            stride=self.dec_str,
        )

        in_shape = self.dec_conv_transpose_input_size(
            out_shape=out_shape, in_channels=in_channels
        )
        return conv_transpose, in_shape

    def __init__(self, config):
        """
        Initialization of the encoder.

        Parameters
        ----------
        config : dic, principal constants to build a encoder.

        Returns
        -------
        None.

        """
        super(DecoderConv, self).__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]
        self.with_sigmoid = config["with_sigmoid"]
        self.img_shape = config["img_shape"]
        self.conv_dim = config["conv_dim"]
        self.n_blocks = config["n_dec_lay"]
        self.dec_c = config["dec_c"]
        self.dec_ks = config["dec_ks"]
        self.dec_str = config["dec_str"]
        self.dec_pad = config["dec_pad"]
        self.dec_dil = config["dec_dil"]
        self.conv_dim = config["conv_dim"]
        self.skip_z = config["skip_z"]
        self.nn_conv_transpose = NN_CONV_TRANSPOSE[self.conv_dim]
        self.nn_batch_norm = NN_BATCH_NORM[self.conv_dim]

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # decoder - layers in reverse order
        conv_transpose_recon, required_in_shape_r = self.end_block(
            out_shape=self.img_shape, dec_c_factor=2 ** (self.n_blocks - 1)
        )
        conv_transpose_scale, required_in_shape_s = self.end_block(
            out_shape=self.img_shape, dec_c_factor=2 ** (self.n_blocks - 1)
        )

        self.conv_transpose_recon = conv_transpose_recon
        self.conv_transpose_scale = conv_transpose_scale

        if np.all(required_in_shape_r != required_in_shape_s):
            raise ValueError
        required_in_shape = required_in_shape_r

        blocks_reverse = torch.nn.ModuleList()
        for i in reversed(range(self.n_blocks - 1)):
            dec_c_factor = 2 ** i

            batch_norm, conv_tranpose, in_shape = self.block(
                out_shape=required_in_shape, dec_c_factor=dec_c_factor
            )

            shape_h = required_in_shape[1] * required_in_shape[2] * dec_c_factor * 2
            w_z = nn.Linear(self.latent_dim, shape_h, bias=False)

            blocks_reverse.append(w_z)
            blocks_reverse.append(batch_norm)
            blocks_reverse.append(conv_tranpose)

            required_in_shape = in_shape

        self.blocks = blocks_reverse[::-1]
        self.in_shape = required_in_shape

        self.fcs_infeatures = functools.reduce((lambda x, y: x * y), self.in_shape)

        self.l0 = nn.Linear(
            in_features=self.latent_dim, out_features=self.fcs_infeatures
        )

    def forward(self, z):
        """
        Compute the passage through the neural network

        Parameters
        ----------
        z : tensor, latent space.

        Returns
        -------
        recon : tensor, image or voxel.
        scale_b: tensor, image or voxel.

        """
        h1 = self.relu(self.l0(z))
        h = h1.view((-1,) + self.in_shape)

        for i in range(self.n_blocks - 1):
            h = self.blocks[3 * i](h)
            h = self.blocks[3 * i + 1](h)
            if self.skip_z:
                z1 = self.blocks[3 * i + 2](z).reshape(h.shape)
                h = self.leakyrelu(h + z1)

        recon = self.conv_transpose_recon(h)
        scale_b = self.conv_transpose_scale(h)

        if self.with_sigmoid:
            recon = self.sigmoid(recon)
        return recon, scale_b


class VaeConv(nn.Module):
    """This class compute the VAE"""

    def __init__(self, config):
        """
        Initialization of the VAE.

        Parameters
        ----------
        config : dic, principal constants to build a encoder.

        Returns
        -------
        None.

        """
        super(VaeConv, self).__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]
        self.img_shape = config["img_shape"]
        self.conv_dim = config["conv_dim"]
        self.with_sigmoid = config["with_sigmoid"]
        self.n_encoder_blocks = config["n_enc_lay"]
        self.n_decoder_blocks = config["n_dec_lay"]

        self.encoder = EncoderConv(config)

        self.decoder = DecoderConv(config)

    def forward(self, x):
        """
        Compute the passage through the neural network

        Parameters
        ----------
        x : tensor, image or voxel.

        Returns
        -------
        res : tensor, image or voxel.
        scale_b: tensor, image or voxel.
        mu : tensor, latent space mu.
        logvar : tensor, latent space sigma.

        """
        mu, logvar, z, matrix = self.encoder(x)
        res, scale_b = self.decoder(z)
        return res, scale_b, mu, logvar, z


def reparametrize(mu, logvar, n_samples=1):
    """
    Transform the probabilistic latent space into a deterministic latent space

    Parameters
    ----------
    mu : tensor, latent space mu.
    logvar : tensor, latent space sigma.
    n_samples : int, number of samples.

    Returns
    -------
    z_flat : tensor, deterministic latent space

    """
    n_batch_data, latent_dim = mu.shape

    std = logvar.mul(0.5).exp_()
    std_expanded = std.expand(n_samples, n_batch_data, latent_dim)
    mu_expanded = mu.expand(n_samples, n_batch_data, latent_dim)

    if CUDA:
        eps = torch.cuda.FloatTensor(n_samples, n_batch_data, latent_dim).normal_()
    else:
        eps = torch.FloatTensor(n_samples, n_batch_data, latent_dim).normal_()
    eps = torch.autograd.Variable(eps)

    z = eps * std_expanded + mu_expanded
    z_flat = z.reshape(n_samples * n_batch_data, latent_dim)
    z_flat = z_flat.squeeze(dim=1)
    return z_flat


def sample_from_q(mu, logvar, n_samples=1):
    """
    Transform a probabilistic latent space into a deterministic latent space

    Parameters
    ----------
    mu : tensor, latent space mu.
    logvar : tensor, latent space sigma.
    n_samples : int, number of samples.

    Returns
    -------
    tensor, deterministic latent space.

    """
    return reparametrize(mu, logvar, n_samples)


def sample_from_prior(latent_dim, n_samples=1):
    """
    Transform a probabilistic latent space into a deterministic latent.

    Parameters
    ----------
    latent_dim : int, latent dimension.
    n_samples : int, optional, number of sample.

    Returns
    -------
    tensor, deterministic latent space.

    """
    if CUDA:
        mu = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
        logvar = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
    else:
        mu = torch.zeros(n_samples, latent_dim)
        logvar = torch.zeros(n_samples, latent_dim)
    return reparametrize(mu, logvar)


class Discriminator(nn.Module):
    """This class compute the GAN"""

    def dis_conv_output_size(self, in_shape, out_channels):
        """
        Compute the output shape by knowing the input shape of a layer.

        Parameters
        ----------
        in_shape : tuple, shape of the input of the layer.
        out_channels : int, number of output channels.
        Returns
        -------
        tuple, shape of the output of the layer.

        """
        return conv_output_size(
            in_shape,
            out_channels,
            kernel_size=self.dis_ks,
            stride=self.dis_str,
            padding=self.dis_pad,
            dilation=self.dis_dil,
        )

    def __init__(self, config):
        """
        Initialization of the GAN.

        Parameters
        ----------
        config : dic, principal constants to build a encoder.

        Returns
        -------
        None.

        """
        super(Discriminator, self).__init__()
        self.img_shape = config["img_shape"]
        self.dis_c = config["dis_c"]
        self.dis_ks = config["dis_ks"]
        self.dis_str = config["dis_str"]
        self.dis_pad = config["dis_pad"]
        self.dis_dil = config["dis_dil"]
        self.n_blocks = config["n_gan_lay"]
        self.config = config
        self.batch_size = config["batch_size"]
        self.latent_dim = config["latent_dim"]
        self.img_shape = config["img_shape"]
        self.conv_dim = config["conv_dim"]
        self.nn_conv = NN_CONV[self.conv_dim]
        self.nn_batch_norm = NN_BATCH_NORM[self.conv_dim]
        self.batch_size = config["batch_size"]

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.blocks = torch.nn.ModuleList()
        self.dis_out_shape = self.img_shape
        next_in_channels = self.img_shape[0]
        for i in range(self.n_blocks):
            dis_c_factor = 2 ** i
            dis = self.nn_conv(
                in_channels=next_in_channels,
                out_channels=self.dis_c * dis_c_factor,
                kernel_size=self.dis_ks,
                stride=self.dis_str,
                padding=self.dis_pad,
            )
            bn = self.nn_batch_norm(dis.out_channels)
            self.blocks.append(dis)
            self.blocks.append(bn)
            next_in_channels = dis.out_channels
            self.dis_out_shape = self.dis_conv_output_size(
                in_shape=self.dis_out_shape, out_channels=next_in_channels
            )

        self.fcs_infeatures = functools.reduce((lambda x, y: x * y), self.dis_out_shape)

        self.fc1 = nn.Linear(in_features=self.fcs_infeatures, out_features=1)

        # # discriminator
        # self.dis1 = self.nn_conv(
        #     in_channels=self.img_shape[0],
        #     out_channels=self.dis_c,
        #     kernel_size=self.dis_ks,
        #     stride=self.dis_str,
        #     padding=self.dis_pad)
        # self.bn1 = self.nn_batch_norm(self.dis1.out_channels)
        # self.dis1_out_shape = self.dis_conv_output_size(
        #     in_shape=self.img_shape,
        #     out_channels=self.dis1.out_channels)

        # self.dis2 = self.nn_conv(
        #     in_channels=self.dis1.out_channels,
        #     out_channels=self.dis_c * 2,
        #     kernel_size=self.dis_ks,
        #     stride=self.dis_str,
        #     padding=self.dis_pad)
        # self.bn2 = self.nn_batch_norm(self.dis2.out_channels)
        # self.dis2_out_shape = self.dis_conv_output_size(
        #     in_shape=self.dis1_out_shape,
        #     out_channels=self.dis2.out_channels)

        # self.dis3 = self.nn_conv(
        #     in_channels=self.dis2.out_channels,
        #     out_channels=self.dis_c * 4,
        #     kernel_size=self.dis_ks,
        #     stride=self.dis_str,
        #     padding=self.dis_pad)
        # self.bn3 = self.nn_batch_norm(self.dis3.out_channels)
        # self.dis3_out_shape = self.dis_conv_output_size(
        #     in_shape=self.dis2_out_shape,
        #     out_channels=self.dis3.out_channels)

        # self.dis4 = self.nn_conv(
        #     in_channels=self.dis3.out_channels,
        #     out_channels=self.dis_c * 8,
        #     kernel_size=self.dis_ks,
        #     stride=self.dis_str,
        #     padding=self.dis_pad)
        # self.bn4 = self.nn_batch_norm(self.dis4.out_channels)
        # self.dis4_out_shape = self.dis_conv_output_size(
        #     in_shape=self.dis3_out_shape,
        #     out_channels=self.dis4.out_channels)

        # self.fcs_infeatures = functools.reduce(
        #     (lambda x, y: x * y), self.dis4_out_shape)

        # self.fc1 = nn.Linear(in_features=self.fcs_infeatures,
        #                      out_features=1)

    def forward(self, x):
        """
        Forward pass of the discriminator is to take an image
        and output probability of the image being generated by the prior
        versus the learned approximation of the posterior.
        Parameters
        ----------
        x : tensor, image or voxel.

        Returns
        -------
        prob: float, between 0 and 1 the probability of being a true image.
        """
        h = x
        for i in range(self.n_blocks):
            h = self.leakyrelu(self.blocks[2 * i + 1](self.blocks[2 * i](h)))
        h = h.view(-1, self.fcs_infeatures)
        h_feature = self.fc1(h)
        prob = self.sigmoid(h_feature)
        prob = prob.view(-1, 1)

        # h1 = self.leakyrelu(self.bn1(self.dis1(x)))
        # h2 = self.leakyrelu(self.bn2(self.dis2(h1)))
        # h3 = self.leakyrelu(self.bn3(self.dis3(h2)))
        # h4 = self.leakyrelu(self.bn4(self.dis4(h3)))
        # h5 = h4.view(-1, self.fcs_infeatures)
        # h5_feature = self.fc1(h5)
        # prob = self.sigmoid(h5_feature)
        # prob = prob.view(-1, 1)

        return prob, 0, 0  # h5_feature,  h5_logvar
