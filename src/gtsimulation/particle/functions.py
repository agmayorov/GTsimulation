import numpy as np


def ConvertR2T(R, M, A, Z):
    # Converts rigidity into kinetic energy per nucleon
    # INPUT:
    #       R - rigidity in GV
    #       M - Mass of particle in GV/c^2
    #       A - Number of nucleons
    #       Z - Charge
    # OUTPUT:
    #       E - Kinetic energy per nucleon in GeV/n

    A = np.array(A)
    A[A == 0] = 1

    E = (1 / A) * (np.sqrt(np.power(Z * R, 2) + np.power(M, 2)) - M)

    return E


def ConvertT2R(T_per_nucleon, M, A, Z):
    # Converts kinetic energy per nucleon into rigidity
    # INPUT:
    #       T_per_nucleon - Kinetic energy per nucleon in GeV/n
    #       M - Mass of particle in GV/c^2
    #       A - Number of nucleons
    #       Z - Charge
    # OUTPUT:
    #       R - Rigidity in GV
    A = np.array(A)
    A[A == 0] = 1

    R = (1 / Z) * (np.sqrt(np.power(A * T_per_nucleon + M, 2) - np.power(M, 2)))

    return R


def convert_units(energy, from_units, to_units, m, a, z):
    valid_units = ('T', 'E', 'R')
    if from_units not in valid_units:
        raise ValueError(f"Unsupported source energy units {from_units!r}; expected one of {valid_units}")
    if to_units not in valid_units:
        raise ValueError(f"Unsupported target energy units {to_units!r}; expected one of {valid_units}")
    if from_units == to_units:
        return np.array(energy, copy=True)

    energy_converted = np.zeros_like(energy)
    match from_units:
        case 'T':
            match to_units:
                case 'R':
                    energy_converted = ConvertT2R(energy, m, a, z)
                case 'E':
                    energy_converted = energy * a
        case 'R':
            match to_units:
                case 'T':
                    energy_converted = ConvertR2T(energy, m, a, z)
                case 'E':
                    energy_converted = ConvertR2T(energy, m, a, z) * a
        case 'E':
            match to_units:
                case 'T':
                    energy_converted = energy / a
                case 'R':
                    energy_converted = ConvertT2R(energy / a, m, a, z)

    return energy_converted


def GetAntiParticle(particle):
    particle.Z *= -1
    particle.PDG *= -1

    if particle.Name.startswith('anti_'):
        particle.Name = particle.Name[5:]
    else:
        particle.Name = f'anti_{particle.Name}'
