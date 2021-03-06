import numpy as np
import operator
import torch
import functools
from functools import reduce
import torch.nn as nn


OUT_PAD = 0

NN_CONV = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d}
NN_CONV_TRANSPOSE = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d}

NN_BATCH_NORM = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d}


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


def conv_transpose_input_size(conv_dim, out_shape, in_channels, kernel_size, stride,
                              padding, dilation, output_padding=OUT_PAD):
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
    kernel_size, stride, padding, dilation = conv_parameters(
        conv_dim, kernel_size, stride, padding, dilation)
    if type(output_padding) is int:
        output_padding = np.repeat(output_padding, conv_dim)

    def one_dim(i_dim):
        """Inverts the formula giving the output shape."""
        shape_i_dim = (
            ((out_shape[i_dim+1]
              + 2 * padding[i_dim]
              - dilation[i_dim] * (kernel_size[i_dim] - 1)
              - output_padding[i_dim] - 1)
             // stride[i_dim])
            + 1)

        if shape_i_dim % 1 != 0:
            raise ValueError
        return int(shape_i_dim)

    in_shape = [one_dim(i_dim) for i_dim in range(conv_dim)]
    in_shape = tuple(in_shape)

    return (in_channels,) + in_shape


def conv_output_size(conv_dim, in_shape, out_channels, kernel_size, stride, padding,
                     dilation):
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
        conv_dim=conv_dim,
        out_shape=in_shape,
        in_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=0,
        dilation=dilation)
    # out_shape = (out_shape[0], out_shape[1], out_shape[2])
    return out_shape


class CompressionConv(nn.Module):
    """Encode images"""

    def __init__(self, config):
        super().__init__()
        self.conv_dim = config["dimension"]
        self.enc_c = config["enc_c"]
        self.enc_ks = config["enc_ks"]
        self.enc_str = config["enc_str"]
        self.enc_pad = config["enc_pad"]
        self.enc_dil = config["enc_dil"]
        self.img_shape = config["img_shape"]
        self.nn_batch_norm = NN_BATCH_NORM[self.conv_dim]
        self.n_blocks = config["n_enc_lay"]
        self.nn_conv = NN_CONV[self.conv_dim]
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.construct_cnn()

    def construct_cnn(self):
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
                padding=self.enc_pad)
            bn = self.nn_batch_norm(enc.out_channels)
            self.blocks.append(enc)
            self.blocks.append(bn)
            enc_out_shape = self.enc_conv_output_size(in_shape=next_in_shape,
                                                      out_channels=enc.out_channels)
            next_in_shape = enc_out_shape
            next_in_channels = enc.out_channels
        self.last_out_shape = next_in_shape
        self.fcs_infeatures = functools.reduce(
            (lambda x, y: x * y), self.last_out_shape)

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
            self.conv_dim,
            in_shape, out_channels,
            kernel_size=self.enc_ks,
            stride=self.enc_str,
            padding=self.enc_pad,
            dilation=self.enc_dil)

    def forward(self, h, n=1):
        """
        Compute the passage through the neural network

        Parameters
        ----------
        h : tensor, image or voxel.

        Returns
        -------
        mu : tensor, latent space mu.
        logvar : tensor, latent space sigma.

        """
        h_shape = h.shape
        for i in range(self.n_blocks):
            h = self.blocks[2*i](h)
            h = self.blocks[2*i+1](h)
            h = self.leakyrelu(h)
        h = h.view(h_shape[0], -1)
        return h


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
            conv_dim=self.conv_dim,
            out_shape=out_shape,
            in_channels=in_channels,
            kernel_size=self.dec_ks,
            stride=self.dec_str,
            padding=self.dec_pad,
            dilation=self.dec_dil)

    def init_latent_space_dim(self, config):
        if config["latent_space"] == "so2":
            return 2
        if config["latent_space"] == "so3":
            return (config["wigner_dim"]+1)**2*config["rep_copies"]
        if config["latent_space"] == "rl":
            return config["latent_dim"]
        if config["latent_space"] == "r1":
            return 1
        if config["latent_space"] == "r2":
            return 2

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

        batch_norm = self.nn_batch_norm(
            num_features=out_channels,
            eps=1.e-3)

        conv_transpose = self.nn_conv_transpose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.dec_ks,
            stride=self.dec_str)

        in_shape = self.dec_conv_transpose_input_size(
            out_shape=out_shape,
            in_channels=in_channels)
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
            stride=self.dec_str)

        in_shape = self.dec_conv_transpose_input_size(
            out_shape=out_shape,
            in_channels=in_channels)
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
        self.latent_dim = self.init_latent_space_dim(self.config)
        self.with_sigmoid = config["with_sigmoid"]
        self.img_shape = config["img_shape"]
        self.conv_dim = config["dimension"]
        self.n_blocks = config["n_dec_lay"]
        self.dec_c = config["dec_c"]
        self.dec_ks = config["dec_ks"]
        self.dec_str = config["dec_str"]
        self.dec_pad = config["dec_pad"]
        self.dec_dil = config["dec_dil"]
        self.skip_z = config["skip_z"]
        self.nn_conv_transpose = NN_CONV_TRANSPOSE[self.conv_dim]
        self.nn_batch_norm = NN_BATCH_NORM[self.conv_dim]

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # decoder - layers in reverse order
        conv_transpose_recon, required_in_shape_r = self.end_block(
            out_shape=self.img_shape, dec_c_factor=2 ** (self.n_blocks-1))
        conv_transpose_scale, required_in_shape_s = self.end_block(
            out_shape=self.img_shape, dec_c_factor=2 ** (self.n_blocks-1))

        self.conv_transpose_recon = conv_transpose_recon
        self.conv_transpose_scale = conv_transpose_scale

        if np.all(required_in_shape_r != required_in_shape_s):
            raise ValueError
        required_in_shape = required_in_shape_r

        blocks_reverse = torch.nn.ModuleList()
        for i in reversed(range(self.n_blocks-1)):
            dec_c_factor = 2 ** i

            batch_norm, conv_tranpose, in_shape = self.block(
                out_shape=required_in_shape,
                dec_c_factor=dec_c_factor)
            shape_h = reduce(operator.mul, required_in_shape, 1)
            w_z = nn.Linear(self.latent_dim, shape_h, bias=False)

            blocks_reverse.append(w_z)
            blocks_reverse.append(batch_norm)
            blocks_reverse.append(conv_tranpose)

            required_in_shape = in_shape

        self.blocks = blocks_reverse[::-1]
        self.in_shape = required_in_shape

        self.fcs_infeatures = functools.reduce(
            (lambda x, y: x * y), self.in_shape)
        self.l0 = nn.Linear(
            in_features=self.latent_dim, out_features=self.fcs_infeatures)

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

        for i in range(self.n_blocks-1):
            h = self.blocks[3*i](h)
            h = self.blocks[3*i+1](h)
            if self.skip_z:
                z1 = self.blocks[3*i+2](z).reshape(h.shape)
                h = self.leakyrelu(h+z1)

        recon = self.conv_transpose_recon(h)
        scale_b = self.conv_transpose_scale(h)

        if self.with_sigmoid:
            recon = self.sigmoid(recon)
        return recon, scale_b
