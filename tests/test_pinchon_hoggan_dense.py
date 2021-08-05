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
        assert matrix.shape == 3
        assert type(matrix) == torch.Tensor
