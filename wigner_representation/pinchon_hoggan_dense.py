"""Compute wigner matrix for neural network by adapting Taco Cohen's paper."""
import os

import numpy as np
import torch

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


def open_Jd():
    """
    Open the file containing the precompute symmetric orthogonal block matrices
    that exchange the Y and Z axis.

    Returns
    -------
    J : list,
        block matrices that exchanges the Y and Z.

    """
    base = "J_dense_0-150.npy"
    path = os.path.join(os.path.dirname(__file__), base)
    Jd_numpy = np.load(path, allow_pickle=True)
    J = []
    for i in range(150):
        J.append(torch.Tensor(Jd_numpy[i]))
    return J


Jd = open_Jd()


def z_rot_mat(angle, la):
    """
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * la + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Parameters
    ----------
    angle : float,
        angle of the rotation around z axis.
    la : int,
        2*la+1 is the dimension of the matrix.

    Returns
    -------
    M : tensor,
        matrix representation of a z-axis rotation.

    """
    M = torch.zeros((2 * la + 1, 2 * la + 1)).to(DEVICE)
    inds = torch.arange(0, 2 * la + 1, 1).to(DEVICE)
    reversed_inds = torch.arange(2 * la, -1, -1).to(DEVICE)
    frequencies = torch.arange(la, -la - 1, -1).to(DEVICE)
    M[inds, reversed_inds] = torch.sin(frequencies * angle).to(DEVICE)
    M[inds, inds] = torch.cos(frequencies * angle).to(DEVICE)
    return M


def rot_mat(alpha, beta, gamma, la, J):
    """
    Compute the representation matrix of a rotation by ZYZ-Euler
    angles (alpha, beta, gamma) in representation l in the basis
    of real spherical harmonics.

    Parameters
    ----------
    alpha : float,
        angle of the rotation around z axis.
    beta : float
        angle of the rotation around y axis.
    gamma : float
        angle of the rotation around z axis.
    la : int,
        2*la+1 is the dimension of the matrix.
    J : tensor
        block matrix that exchanges the Y and Z axis.


    Returns
    -------
    matrix : tensor
        representation matrix of a rotation.

    """

    Xa = z_rot_mat(alpha, la)
    Xb = z_rot_mat(beta, la)
    Xc = z_rot_mat(gamma, la)
    m1 = torch.matmul(torch.matmul(torch.matmul(Xa, J), Xb), J)
    matrix = torch.matmul(m1, Xc)
    return matrix


def derivative_z_rot_mat(angle, la):
    """
    Compute the derivative matrix representation of a z-axis rotation by a
    given angle

    Parameters
    ----------
    angle : float,
        angle of the rotation around z axis.
    la : int,
        2*la+1 is the dimension of the matrix.

    Returns
    -------
    M : tensor,
        derivative matrix representation of a z-axis rotation.

    """
    M = np.zeros((2 * la + 1, 2 * la + 1))
    inds = np.arange(0, 2 * la + 1, 1)
    reversed_inds = np.arange(2 * la, -1, -1)
    frequencies = np.arange(la, -la - 1, -1)
    M[inds, reversed_inds] = np.cos(frequencies * angle) * frequencies
    M[inds, inds] = -np.sin(frequencies * angle) * frequencies
    return M


def derivative_rot_mat(alpha, beta, gamma, la, J):
    """
    Compute the derivative representation matrix of a rotation by ZYZ-Euler
    angles (alpha, beta, gamma) in representation l in the basis
    of real spherical harmonics.

    Parameters
    ----------
    alpha : float,
        angle of the rotation around z axis.
    beta : float
        angle of the rotation around y axis.
    gamma : float
        angle of the rotation around z axis.
    la : int,
        2*la+1 is the dimension of the matrix.
    J : tensor
        block matrix that exchanges the Y and Z axis.


    Returns
    -------
    matrix : tensor
        derivative matrix representation of a rotation.

    """
    Xa = z_rot_mat(alpha, la)
    Xb = z_rot_mat(beta, la)
    Xc = z_rot_mat(gamma, la)
    dXa_da = derivative_z_rot_mat(alpha, la)
    dXb_db = derivative_z_rot_mat(beta, la)
    dXc_dc = derivative_z_rot_mat(gamma, la)

    dDda = dXa_da.dot(J).dot(Xb).dot(J).dot(Xc)
    dDdb = Xa.dot(J).dot(dXb_db).dot(J).dot(Xc)
    dDdc = Xa.dot(J).dot(Xb).dot(J).dot(dXc_dc)
    return dDda, dDdb, dDdc
