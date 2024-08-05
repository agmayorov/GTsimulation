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
    # Converts particles energy into rigidity
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

def ConvertUnits(Energy, FromUnits, ToUnits, M, A, Z):
    match FromUnits:
        case 'T':
            match ToUnits:
                case 'R':
                    EnergyConverted = ConvertT2R(Energy, M, A, Z)
                case 'E':
                    EnergyConverted = Energy * A
        case 'R':
            match ToUnits:
                case 'T':
                    EnergyConverted = ConvertR2T(Energy, M, A, Z)
                case 'E':
                    EnergyConverted = ConvertR2T(Energy, M, A, Z) * A
        case 'E':
            match ToUnits:
                case 'T':
                    EnergyConverted = Energy / A
                case 'R':
                    EnergyConverted = ConvertT2R(Energy / A, M, A, Z)

    return EnergyConverted


def GetAntiParticle(particle):
    particle.Z *= -1
    particle.PDG *= -1

    if particle.Name[0] == 'a':
        particle.Name = particle.Name[1:]
    else:
        particle.Name = 'a' + particle.Name