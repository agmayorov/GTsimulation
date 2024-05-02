from Particle.NucleiProp import NucleiProp


def GetNucleiProp(T):
    # The function returns the particle /nucleus properties.
    #   Ver. 1, red. 2 / January 2022 / A. Mayorov
    #
    #   Arguments:
    #       T  -  String  -  Name of the particle or nucleus (see table below)
    #   Output:
    #       A  -  Int     -  Mass number (number of nucleons)
    #       Z  -  Int     -  Charge (number of protons)
    #       M  -  Float   -  Mas in MeV
    #     PDG  -          -  Particle Data Group number
    #      Ab  -  Float   -  Abundance in nature (relative to isotope)
    #     Thd  -  Float   -  Time of half decay (for unstable isotopes), years
    # Examples:
    # A, Z, M, PDG, Ab, Thd = GetNucleiProp('tr')
    # A, Z, M, PDG, Ab, Thd = GetNucleiProp('Be-10')

    if T in NucleiProp:
        A = NucleiProp[T]['A']
        Z = NucleiProp[T]['Z']
        M = NucleiProp[T]['M']
        PDG = NucleiProp[T]['PDG']
        if 'Ab' not in NucleiProp[T] or 'Thd' not in NucleiProp[T]:
            Ab = None
            Thd = None
        else:
            Ab = NucleiProp[T]['Ab']
            Thd = NucleiProp[T]['Thd']
    else:
        A, Z, M, Ab, Thd = -1
        PDG = 0

    return A, Z, M, PDG, Ab, Thd
