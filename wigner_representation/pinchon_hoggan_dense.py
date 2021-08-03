import os

import numpy as np
import torch
from scipy.linalg import block_diag

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

base = "J_dense_0-150.npy"
path = os.path.join(os.path.dirname(__file__), base)
Jd = np.load(path, allow_pickle=True)
J = []
for i in range(150):
    J.append(torch.Tensor(Jd[i]))
Jd = J.copy()


def SO3_irreps(g, irreps):
    global Jd

    # First, compute sinusoids at all required frequencies, i.e.
    # cos(n x) for n=0, ..., max(irreps)
    # sin(n x) for n=-max(irreps), ..., max(irreps)
    # where x ranges over the three parameters of SO(3).

    # In theory, it may be faster to evaluate cos(x) once and then use
    # Chebyshev polynomials to obtain cos(n*x), but in practice this appears
    # to be slower than just evaluating cos(n*x).
    dim = np.sum(2 * np.array(irreps) + 1)
    T = np.empty((dim, dim, g.shape[1]))
    lenght = g.shape[1]
    for i in range(lenght):
        T[:, :, i] = block_diag(
            *[rot_mat(g[0, i], g[1, i], g[2, i], la, Jd[la]) for la in irreps]
        )
    return T


def SO3_irrep(g, la):
    global Jd
    g = np.atleast_2d(g)
    T = np.empty((2 * la + 1, 2 * la + 1, g.shape[1]))
    for i in range(g.shape[1]):
        T[:, :, i] = rot_mat(g[0, i], g[1, i], g[2, i], la, Jd[la])
    return T  # np.squeeze(T)


def z_rot_mat(angle, la):
    """
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
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

    The result is the same as the wignerD_mat function by Johann Goetz,
    when the sign of alpha and gamma is flipped.

    The forementioned function is here:
    https://sites.google.com/site/theodoregoetz/notes/wignerdfunction
    """
    Xa = z_rot_mat(alpha, la)
    Xb = z_rot_mat(beta, la)
    Xc = z_rot_mat(gamma, la)
    m1 = torch.matmul(torch.matmul(torch.matmul(Xa, J), Xb), J)
    matrix = torch.matmul(m1, Xc)
    return matrix


def derivative_z_rot_mat(angle, la):
    M = np.zeros((2 * la + 1, 2 * la + 1))
    inds = np.arange(0, 2 * la + 1, 1)
    reversed_inds = np.arange(2 * la, -1, -1)
    frequencies = np.arange(la, -la - 1, -1)
    M[inds, reversed_inds] = np.cos(frequencies * angle) * frequencies
    M[inds, inds] = -np.sin(frequencies * angle) * frequencies
    return M


def derivative_rot_mat(alpha, beta, gamma, la, J):
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
