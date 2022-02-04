import torch
import torch.nn as nn
import lie_tools


class ActionNetSo3(nn.Module):
    """Uses proper group action."""

    def __init__(self, degrees, rep_copies=10,
                 with_mlp=False, item_rep=None, transpose=False):
        """Action decoder.
        Params:
        - degrees : max number of degrees of representation,
                    harmonics matrix has (degrees+1)^2 rows
        - deconv : deconvolutional network used after transformation
        - content_dims : content vector dimension
        - rep_copies : number of copies of representation / number of dimension
                       of signal on sphere / columns of harmonics matrix
        - harmonics_encoder_layers : number of layers of MLP that transforms
                                     content vector to harmonics matrix
        - with_mlp : route transformed harmonics through MLP before deconv
        - item_rep : optional fixed single item rep
        - transpose : Whether to take transpose of fourier matrices
        """
        super().__init__()
        self.degrees = degrees
        self.rep_copies = rep_copies
        self.matrix_dims = (degrees + 1) ** 2
        self.transpose = transpose

        if item_rep is None:
            self.item_rep = nn.Parameter(
                torch.randn((self.matrix_dims, rep_copies)))
        else:
            self.register_buffer('item_rep', item_rep)

    def forward(self, angles):
        """Input is ZYZ Euler angles and possibly content vector."""
        n, d = angles.shape

        assert d == 3, 'Input should be Euler angles.'

        harmonics = self.item_rep.expand(n, -1, -1)
        item = lie_tools.block_wigner_matrix_multiply(
            angles, harmonics, self.degrees, transpose=self.transpose) \
            .view(-1, self.matrix_dims * self.rep_copies)

        return item


class ActionNetSo2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, angles):
        n, d = angles.shape
        if d == 1:
            item = torch.cat([torch.cos(angles), -torch.sin(angles)], 1)
            return item
        return nn.functional.normalize(angles, eps=1e-30)


class ActionNetRL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, latent_space):
        return latent_space
