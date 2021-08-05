"""Test pinchon_hoggan_dense"""

import torch
from wigner_representation import pinchon_hoggan_dense as phd


class TestPinchon:
    @staticmethod
    def test_open_Jd():
        Jd = phd.open_Jd()

        assert type(Jd[0]) is torch.Tensor
        assert len(Jd) == 150

    @staticmethod
    def test_z_rot_mat():
        angle = 1
        la = 1
        matrix = phd.z_rot_mat(angle, la)
        assert matrix.shape == (3, 3)
        assert type(matrix) == torch.Tensor

    @staticmethod
    def test_rot_mat():
        alpha = 1
        beta = 1
        gamma = 1
        la = 2
        Jd = phd.open_Jd()
        J = Jd[la]
        matrix = phd.rot_mat(alpha, beta, gamma, la, J)
        assert matrix.shape == (5, 5)
