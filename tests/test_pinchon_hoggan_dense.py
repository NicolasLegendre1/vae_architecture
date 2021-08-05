"""Test pinchon_hoggan_dense"""
import numpy as np
import torch

from wigner_representation import pinchon_hoggan_dense as phd


class TestPinchon:
    @staticmethod
    def test_open_Jd():
        Jd = phd.open_Jd()

        assert type(Jd[0]) is torch.Tensor
        assert len(Jd) == 150
        assert 1 == 1
