import numpy as np


def ConvertR2T(R, M, A, Z):
    # Converts rigidity into Energy
    # INPUT:
    #       R - rigidity in GV
    #       M - Mass of particle in GV/c^2
    #       A - Number of nuclons
    #       Z - Charge
    # OUTPUT:
    #       E - Kinetic energy in GeV

    A = np.array(A)
    A[A == 0] = 1

    E = (1 / A) * (np.sqrt(np.power(Z * R, 2) + np.power(M, 2)) - M)

    return E


def ConvertT2R(T, M, A, Z):
    # Converts particles energy into  rigidity
    # INPUT:
    #       T - Kinetic energy in GeV
    #       M - Mass of particle in GV/c^2
    #       A - Number of nuclons
    #       Z - Charge
    # OUTPUT:
    #       R - Rigidity in GV
    A = np.array(A)
    A[A == 0] = 1

    R = (1 / Z) * (np.sqrt(np.power(A * T + M, 2) - np.power(M, 2)))

    return R
