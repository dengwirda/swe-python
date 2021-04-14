
import numpy as np

# Simple inv, det utilities for small-matrices.
# Authors: Darren Engwirda

def det_2x2(amat):

# DET-2x2: compute determinants for a set of 2 x 2 matrices.

    return \
        amat[0, 0, :] * amat[1, 1, :] - \
        amat[0, 1, :] * amat[1, 0, :]


def det_3x3(amat):

# DET-3x3: compute determinants for a set of 3 x 3 matrices.

    detA = \
        amat[1, 1, :] * amat[2, 2, :] - \
        amat[1, 2, :] * amat[2, 1, :]

    detB = \
        amat[1, 0, :] * amat[2, 2, :] - \
        amat[1, 2, :] * amat[2, 0, :]

    detC = \
        amat[1, 0, :] * amat[2, 1, :] - \
        amat[1, 1, :] * amat[2, 0, :]

    return \
        amat[0, 0, :] * detA - \
        amat[0, 1, :] * detB + \
        amat[0, 2, :] * detC


def inv_2x2(amat):

# INV-2x2: calculate inverse(s) for a set of 2 x 2 matrices.

    adet = det_2x2(amat)

    ainv = np.empty(
        (2, 2, amat.shape[2]), dtype=amat.dtype)

    ainv[0, 0, :] = +amat[1, 1, :]
    ainv[1, 1, :] = +amat[0, 0, :]
    ainv[0, 1, :] = -amat[0, 1, :]
    ainv[1, 0, :] = -amat[1, 0, :]

    return ainv, adet


def inv_3x3(amat):

# INV-3x3: calculate inverse(s) for a set of 3 x 3 matrices.

    adet = det_3x3(amat)

    ainv = np.empty(
        (3, 3, amat.shape[2]), dtype=amat.dtype)

    ainv[0, 0, :] = \
        amat[2, 2, :] * amat[1, 1, :] - \
        amat[2, 1, :] * amat[1, 2, :]

    ainv[0, 1, :] = \
        amat[2, 1, :] * amat[0, 2, :] - \
        amat[2, 2, :] * amat[0, 1, :]

    ainv[0, 2, :] = \
        amat[1, 2, :] * amat[0, 1, :] - \
        amat[1, 1, :] * amat[0, 2, :]

    ainv[1, 0, :] = \
        amat[2, 0, :] * amat[1, 2, :] - \
        amat[2, 2, :] * amat[1, 0, :]

    ainv[1, 1, :] = \
        amat[2, 2, :] * amat[0, 0, :] - \
        amat[2, 0, :] * amat[0, 2, :]

    ainv[1, 2, :] = \
        amat[1, 0, :] * amat[0, 2, :] - \
        amat[1, 2, :] * amat[0, 0, :]

    ainv[2, 0, :] = \
        amat[2, 1, :] * amat[1, 0, :] - \
        amat[2, 0, :] * amat[1, 1, :]

    ainv[2, 1, :] = \
        amat[2, 0, :] * amat[0, 1, :] - \
        amat[2, 1, :] * amat[0, 0, :]

    ainv[2, 2, :] = \
        amat[1, 1, :] * amat[0, 0, :] - \
        amat[1, 0, :] * amat[0, 1, :]

    return ainv, adet
