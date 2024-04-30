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

    if hasattr(A, '__iter__'):
        for el in A:
            if el == 0:
                el = 1
    elif A == 0:
        A = 1

    E = (1 / A) * (np.sqrt(np.power(Z * R, 2) + np.power(M, 2)) - M)

    return E
