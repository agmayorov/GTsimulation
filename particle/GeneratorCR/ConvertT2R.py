import numpy as np


def ConvertT2R(T, M, A, Z):
    # Converts particles energy into  rigidity
    # INPUT:
    #       T - Kinetic energy in GeV
    #       M - Mass of particle in GV/c^2
    #       A - Number of nuclons
    #       Z - Charge
    # OUTPUT:
    #       R - Rigidity in GV
    if hasattr(A, '__iter__'):
        for el in A:
            if el == 0:
                el = 1
    elif A == 0:
        A = 1
    R = (1 / Z) * (np.sqrt(np.power(A * T + M, 2) - np.power(M, 2)))

    return R
