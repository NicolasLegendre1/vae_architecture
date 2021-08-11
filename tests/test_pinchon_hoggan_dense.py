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
        angle = 0
        la = 1
        matrix = phd.z_rot_mat(angle, la)
        assert matrix.shape == (3, 3)

    @staticmethod
    def test_rot_mat():
        alpha = 0
        beta = 0
        gamma = 0
        la = 2
        Jd = phd.open_Jd()
        J = Jd[la]
        matrix = phd.rot_mat(alpha, beta, gamma, la, J)
        Id = torch.eye(5)
        assert matrix.shape == (5, 5)
        assert (Id == matrix).all()
