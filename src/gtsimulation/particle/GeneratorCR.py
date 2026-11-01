from pathlib import Path

import numpy as np

from gtsimulation.particle._particle import Particle
from gtsimulation.particle.functions import ConvertT2R, ConvertR2T

# Boschini et al. (2020), numerical tables: Z, kinetic energy per nucleon (GeV/n), intensity (m2 s sr GeV/n)^-1
_BOSCHINI2020 = np.loadtxt(Path(__file__).resolve().parent / "data" / "boschini2020_lis_ekin.txt", skiprows=16)

# Bisschoff, Potgieter & Aslam (2019), Eqs. (12)-(18); T in GeV/n, beta2 = beta^2
_BISSCHOFF2019 = {
    "e-": lambda T, beta2: (255.0 / beta2 * T ** -1.0 * ((T + 0.63) / 1.63) ** -2.43
                            + 6.4 * T ** 2.0 * ((T + 15.0) / 16.0) ** -26.0),
    "e+": lambda T, beta2: (25.0 / beta2 * T ** 0.1 * ((T ** 1.1 + 0.2 ** 1.1) / (1 + 0.2 ** 1.1)) ** -3.31
                            + 23.0 * T ** 0.5 * ((T + 2.2) / 3.2) ** -9.5),
    1: lambda T, beta2: (2620.0 / beta2 * T ** 1.1 * ((T ** 0.98 + 0.7 ** 0.98) / (1 + 0.7 ** 0.98)) ** -4.0
                         + 30.0 * T ** 2.0 * ((T + 8.0) / 9.0) ** -12.0),
    2: lambda T, beta2: 163.4 / beta2 * T ** 1.1 * ((T ** 0.97 + 0.58 ** 0.97) / (1 + 0.58 ** 0.97)) ** -4.0,
    5: lambda T, beta2: (1.0 / beta2 * T ** 1.7 * ((T + 0.685) / 1.685) ** -4.8
                         + 3.0e-4 * T ** 3.0 * ((T + 0.204) / 1.204) ** -11.0),
    6: lambda T, beta2: 3.3 / beta2 * T ** 1.22 * ((T ** 0.9 + 0.63 ** 0.9) / (1 + 0.63 ** 0.9)) ** -4.43,
    8: lambda T, beta2: 3.3 / beta2 * T ** 1.23 * ((T ** 0.86 + 0.62 ** 0.86) / (1 + 0.62 ** 0.86)) ** -4.62,
}

# Unpublished fit: J0, a, g2, g3
_CUSTOM_FIT = {
    1: (1872.87554874068, 1.61113357145177, 1.28761628389517, -2.17219702386621),
    2: (139.779118125181, 1.13399984942772, 1.21715858386835, -2.21065644455211),
    3: (0.67360595806831, 1.56511433422331, 1.50073847406430, -2.00837249948625),
    4: (0.33984248416068, 1.52358690395871, 1.24364031903675, -2.45127118367176),
    5: (1.08760559922283, 1.48379096759935, 1.35693206808129, -2.22838803240006),
    6: (3.68261223136148, 1.32184803136140, 1.28478271904066, -2.08425524409385),
    7: (1.01406884838729, 1.47722025525677, 1.40847267712337, -2.00754622517579),
    8: (3.52342241967566, 1.23716432835891, 1.08737530145770, -2.46629906200007),
    9: (0.06742830119460, 1.49951022498618, 1.05482398425009, -2.90115783293288),
    10: (0.53818706959626, 1.51113615332032, 1.22671674504678, -2.20688620129531),
    11: (0.11973060415613, 1.29489790732712, 0.99296113338822, -2.98159372048410),
    12: (0.67412251432422, 1.41840400173818, 1.09661074771491, -2.49504637558846),
    13: (0.12316775376951, 1.42037408858333, 1.35437363481094, -2.03254549530120),
    14: (0.49507921860902, 1.43442256038579, 0.95735286877508, -2.87115816939409),
    26: (0.23975797617632, 1.64447842399986, 0.83030874570426, -3.28332449585784),
    "e-": (36.5063815538979, 2.14364556649091, 2.77366285042517, -1.18498108356250),
    "e+": (6.01437512970430, 1.36377570394312, 6.11040082448321, -0.49390697550654),
    "anti_proton": (0.02021579494618, 6.44997658613249, 2.79960549756288, -1.01848000448823),
}


def GeneratorCR(Source, Spectrum, Particle, Nevents=1, Verbose=0):
    # Example
    # flux = GeneratorCR({'Radius': 10}, {'R': 5}, {'Type': ['pr', 'he4'], 'Abundance': [0.9, 0.1]}, 10)

    # Read input
    # if Verbose == 1:
    #     print('   Reading input ...\n')
    Mode, Ro, Rc = GetSourceArguments(Source)
    EnergyRangeUnits, EnergyRange, SpectrumBase, SpectrumIndex = GetSpectrumArguments(Spectrum)
    Nparticles, ParticleType, Abundance = GetParticleArguments(Particle)

    # if Nparticles > Nevents:
    #     print('    Number of particles > Number of events')
    #     exit(-1)
    #
    # if Verbose == 1:
    #     print('      + Source parameters\n')
    #     print('        Ro = %d\n' % Ro)
    #     print('        Rc = [%.1f, %.1f, %.1f]\n' % (Rc[0], Rc[1], Rc[2]))
    #     print('      + Spectrum\n')
    #     if EnergyRange.size == 1:
    #         print('        EnergyRange = %.1f\n' % EnergyRange)
    #     else:
    #         print('        EnergyRange = [%.1f, %.1f]\n' % (EnergyRange[0], EnergyRange[1]))
    #
    #     print('        EnergyRangeUnits = %s\n' % EnergyRangeUnits)
    #     print('        SpectrumBase = %s\n' % SpectrumBase)
    #     print('        SpectrumIndex = %.1f\n' % SpectrumIndex)
    #     if Nparticles:
    #         print('      + Chemical composition: %d particles\n' % Nparticles)
    #         for s in range(Nparticles):
    #             print('           Particle: %s\n' % list(ParticleType)[s])
    #             print('           Abundance: %.1f\n' % Abundance[s])
    #     else:
    #         print('      + Chemical composition: GCR\n')
    #
    #     print('   Number of events: %d\n' % Nevents)

    # Simulation

    # if Verbose == 1:
    #     print('   Simulation ...\n')
    #     print('      + Coordinates and velocities\n')

    match Mode:
        case 'inward':
            theta = np.acos(1 - 2 * np.random.rand(int(Nevents), 1))
            phi = 2 * np.pi * np.random.rand(int(Nevents), 1)
            r = np.concatenate((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1)

            newZ = r
            newX = np.cross(newZ, np.repeat([[0, 0, 1]], int(Nevents), 0))
            newY = np.cross(newZ, newX)

            S = np.array([np.array([newX[i] / np.linalg.norm(newX[i]), newY[i] / np.linalg.norm(newY[i]), newZ[i] /
                                    np.linalg.norm(newZ[i])]).transpose() for i in range(int(Nevents))])

            ksi = np.random.rand(int(Nevents), 1)
            sin_theta = np.sqrt(ksi)
            cos_theta = np.sqrt(1 - ksi)
            phi = 2 * np.pi * np.random.rand(int(Nevents), 1)
            p = np.concatenate((-sin_theta * np.cos(phi), -sin_theta * np.sin(phi), -cos_theta), axis=1)
            r = r * Ro + Rc
            v = np.array([np.matmul(S[i], p[i]) for i in range(int(Nevents))])

        case 'outward':
            r = np.repeat(Rc, int(Nevents), 0)
            theta = np.acos(1 - 2 * np.random.rand(int(Nevents), 1))
            phi = 2 * np.pi * np.random.rand(int(Nevents), 1)
            v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    # if Verbose == 1:
    #     print('      + Energy spectra\n')

    if Nparticles:
        # User's chemical composition case
        Energy = np.zeros(int(Nevents))
        Nabundance = np.floor(Nevents * np.array(Abundance))
        if sum(Nabundance) != Nevents:
            Nabundance[-1] = Nevents - sum(Nabundance[:-1])
        Nindex = np.concatenate(([0], np.cumsum(Nabundance)), axis=0)
        Nindex = [int(el) for el in Nindex]
        ParticleName = np.concatenate(
            [np.repeat(list(ParticleType)[i], Nabundance[i]) for i in range(len(ParticleType))])
        if len(ParticleName) == 2:
            ParticleName = np.concatenate((ParticleName[0], ParticleName[1]))
        for s in range(int(Nparticles)):
            i1 = Nindex[s]
            i2 = Nindex[s + 1]

            match SpectrumBase:
                case None:
                    # ------- Mono Energy -------
                    Energy[i1: i2] = EnergyRange
                # case 'T' | 'R' | 'E':
                #     # ------- Power Spectrum -------
                #     A, Z, M, *_ = GetNucleiProp(list(ParticleType)[s])
                #     M = M / 1e3  # MeV/c2 -> GeVA, /c2
                #     if EnergyRangeUnits != SpectrumBase:
                #         EnergyRangeS = ConvertUnits(EnergyRange, EnergyRangeUnits, SpectrumBase, M, A, Z)
                #     else:
                #         EnergyRangeS = EnergyRange
                #     ksi = np.random.rand(int(Nabundance[s]))
                #     if SpectrumIndex == -1:
                #         Energy[i1: i2] = EnergyRangeS[0] * np.power((EnergyRangeS[1] / EnergyRangeS[0]), ksi)
                #     else:
                #         g = SpectrumIndex + 1
                #         Energy[i1: i2] = np.power(np.power(EnergyRangeS[0], g) +
                #                                   ksi * (np.power(EnergyRangeS[1], g) -np.power(EnergyRangeS[0], g)),
                #                                   (1 / g))
                #
                #     if EnergyRangeUnits != SpectrumBase:
                #         Energy[i1: i2] = ConvertUnits(Energy[i1: i2], SpectrumBase, EnergyRangeUnits, M, A, Z)
                # case 'F':
                #     # ------- Force-field -------
                #     A, Z, M, *_ = GetNucleiProp(list(ParticleType)[s])
                #     M = M / 1e3  # MeV / c2 -> GeV / c2
                #     if EnergyRangeUnits == 'T':
                #         EnergyRangeS = ConvertUnits(EnergyRange, EnergyRangeUnits, 'T', M, A, Z)
                #     else:
                #         EnergyRangeS = EnergyRange
                #
                #     Jmax = np.max(GetGCRflux('T', np.logspace(np.log10(EnergyRangeS[0]), np.log10(EnergyRangeS[1]),
                #                                               int(1e3)), SpectrumIndex, list(ParticleType)[s]))
                #     iFilled = 0
                #     bunchSize = 1e6  # for faster computing
                #     while True:
                #         Eplayed = EnergyRangeS[0] + np.random.rand(int(bunchSize)) * (EnergyRangeS[1] - EnergyRangeS[0])
                #         ksi = Jmax * np.random.rand(int(bunchSize))
                #
                #         Esuited = Eplayed[ksi < GetGCRflux('T', Eplayed, SpectrumIndex, list(ParticleType)[s])]
                #         if Esuited.size < Nabundance[s] - iFilled:
                #             Energy[i1 + iFilled: i1 + iFilled + Esuited.size - 1] = Esuited
                #             iFilled = iFilled + len(Esuited)
                #         else:
                #             iCut = Nabundance[s] - iFilled
                #             Energy[i1 + iFilled: i2] = Esuited[:int(iCut)]
                #             break
                #     if EnergyRangeUnits != 'T':
                #         Energy[i1: i2] = ConvertUnits(Energy[i1: i2], 'T', EnergyRangeUnits, M, A, Z)
    else:
        # GCR chemical composition case
        ParticleType = ['pr', 'he4', 'Li-7', 'Be-9', 'B-11', 'C-12', 'N-14', 'O-16', 'F-19',
                        'Ne-20', 'Na-23', 'Mg-24', 'Al-27', 'Si-28', 'Fe-56', 'ele', 'pos', 'apr']
        EnergyArgument = {(EnergyRangeUnits + 'min'): EnergyRange[0], (EnergyRangeUnits + 'max'): EnergyRange[1],
                          'Base': 'F', 'Index': SpectrumIndex}
        if Nevents > 1:
            Energy = GeneratorCR({'Radius': 1}, EnergyArgument, {'Type': ['pr', 'he4'],
                                                                 'Abundance': [0.9, 0.1]}, Nevents, 0).E
        else:
            if np.random.rand() < 0.9:
                Energy = GeneratorCR({'Radius': 1}, EnergyArgument, {'Type': ['pr']}, Nevents, 0).E
            else:
                Energy = GeneratorCR({'Radius': 1}, EnergyArgument, {'Type': ['he4']}, Nevents, 0).E
        Jparticle = GetGCRflux(EnergyRangeUnits, Energy, SpectrumIndex)
        iParticle = np.sum(np.random.rand(int(Nevents)) > np.cumsum(Jparticle / np.sum(Jparticle, 0), 0), 1)
        ParticleName = np.array([ParticleType[i] for i in iParticle])

    Energy = np.array(Energy)

    # flux = Flux(r, v, Energy, ParticleName)
    return r, v, Energy, ParticleName
    # return flux


def GetSourceArguments(SourceArguments):
    if 'Mode' in SourceArguments:
        if SourceArguments['Mode'] == 'inward' or SourceArguments['Mode'] == 'outward':
            Mode = SourceArguments['Mode']
        else:
            print('Incorrect mode!')
            exit(-1)
    else:
        Mode = 'inward'

    if 'Radius' in SourceArguments:
        if SourceArguments['Radius'] >= 0:
            Ro = SourceArguments['Radius']
        else:
            print('Incorrect radius!')
            exit(-1)
    else:
        Ro = 1

    if 'Center' in SourceArguments:
        Rc = SourceArguments['Center']
    else:
        Rc = np.zeros(3)

    return Mode, Ro, Rc


def GetSpectrumArguments(SpectrumArguments):
    if 'T' not in SpectrumArguments:
        SpectrumArguments['T'] = 0
    elif SpectrumArguments['T'] <= 0:
        print('Incorrect T value!')
        exit(-1)

    if 'Tmin' not in SpectrumArguments:
        SpectrumArguments['Tmin'] = 0
    elif SpectrumArguments['Tmin'] <= 0:
        print('Incorrect Tmin value!')
        exit(-1)

    if 'Tmax' not in SpectrumArguments:
        SpectrumArguments['Tmax'] = float('inf')
    elif SpectrumArguments['Tmax'] <= 0:
        print('Incorrect Tmax value!')
        exit(-1)

    if 'R' not in SpectrumArguments:
        SpectrumArguments['R'] = 0
    elif SpectrumArguments['R'] <= 0:
        print('Incorrect R value!')
        exit(-1)

    if 'Rmin' not in SpectrumArguments:
        SpectrumArguments['Rmin'] = 0
    elif SpectrumArguments['Rmin'] <= 0:
        print('Incorrect Rmin value!')
        exit(-1)

    if 'Rmax' not in SpectrumArguments:
        SpectrumArguments['Rmax'] = float('inf')
    elif SpectrumArguments['Rmax'] <= 0:
        print('Incorrect Rmax value!')
        exit(-1)

    if 'E' not in SpectrumArguments:
        SpectrumArguments['E'] = 0
    elif SpectrumArguments['E'] <= 0:
        print('Incorrect E value!')
        exit(-1)

    if 'Emin' not in SpectrumArguments:
        SpectrumArguments['Emin'] = 0
    elif SpectrumArguments['Emin'] <= 0:
        print('Incorrect Emin value!')
        exit(-1)

    if 'Emax' not in SpectrumArguments:
        SpectrumArguments['Emax'] = float('inf')
    elif SpectrumArguments['Emax'] <= 0:
        print('Incorrect Emax value!')
        exit(-1)

    if 'Base' not in SpectrumArguments:
        SpectrumArguments['Base'] = None
    elif SpectrumArguments['Base'] not in {'', 'T', 'R', 'E', 'F'}:
        print('Incorrect Base value!')
        exit(-1)

    if 'Index' not in SpectrumArguments:
        SpectrumArguments['Index'] = 1

    caseVector = [SpectrumArguments['T'] > 0,
                  SpectrumArguments['Tmin'] > 0 and SpectrumArguments['Tmax'] < float('inf'),
                  SpectrumArguments['R'] > 0,
                  SpectrumArguments['Rmin'] > 0 and SpectrumArguments['Rmax'] < float('inf'),
                  SpectrumArguments['E'] > 0,
                  SpectrumArguments['Emin'] > 0 and SpectrumArguments['Emax'] < float('inf')]

    if sum(caseVector) != 1:
        print('    The particle energy is incorrectly set')
        exit(-1)

    match (caseVector.index(next(filter(lambda x: x != 0, caseVector)))):
        case 0:
            EnergyRangeUnits = 'T'
            EnergyRange = np.array(SpectrumArguments['T'])
        case 1:
            EnergyRangeUnits = 'T'
            EnergyRange = np.array([SpectrumArguments['Tmin'], SpectrumArguments['Tmax']])
            if SpectrumArguments['Tmax'] < SpectrumArguments['Tmin']:
                exit(-1)
        case 2:
            EnergyRangeUnits = 'R'
            EnergyRange = np.array(SpectrumArguments['R'])
        case 3:
            EnergyRangeUnits = 'R'
            EnergyRange = np.array([SpectrumArguments['Rmin'], SpectrumArguments['Rmax']])
            if SpectrumArguments['Rmax'] < SpectrumArguments['Rmin']:
                exit(-1)
        case 4:
            EnergyRangeUnits = 'E'
            EnergyRange = np.array(SpectrumArguments['E'])
        case 5:
            EnergyRangeUnits = 'E'
            EnergyRange = np.array([SpectrumArguments['Emin'], SpectrumArguments['Emax']])
            if SpectrumArguments['Emax'] < SpectrumArguments['Emin']:
                exit(-1)

    SpectrumBase = SpectrumArguments['Base']
    if not hasattr(EnergyRange, '__iter__') and SpectrumBase == 'F':
        print('    Force-field simulation for mono line can not be done')
        exit(-1)

    SpectrumIndex = SpectrumArguments['Index']
    if SpectrumBase == 'F' and SpectrumIndex < 0:
        print('    Modulation potential should be positive')

    # !!! !!! SpectrumIndex < 0.2 temporary crutch - remove after updating GetGCRGlux !!! !!!
    if SpectrumBase == 'F' and SpectrumIndex < 0.2:
        print('    Modulation potential < 0.2 GV work wrong')

    # !!! !!! SpectrumIndex < 0.2 temporary crutch - remove after updating GetGCRGlux !!! !!!

    return EnergyRangeUnits, EnergyRange, SpectrumBase, SpectrumIndex


def GetParticleArguments(ParticleArguments):
    Nparticles = len(ParticleArguments['Type'])

    ParticleType = ParticleArguments['Type']
    if Nparticles == 1 and list(ParticleType)[0] == 'GCR':
        Nparticles = 0

    if 'Abundance' in ParticleArguments:
        Abundance = ParticleArguments['Abundance']
        if abs(sum(Abundance) - 1) > 1e-16:
            print('    sum(Abundance) should be equal to 1')
            exit(-1)
    elif Nparticles == 1 or Nparticles == 0:
        Abundance = [1]
    else:
        print('Incorrect abundance!')
        exit(-1)

    return Nparticles, ParticleType, Abundance


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


def GetGCRflux(E_type, E, F, PartName=None):
    #   Ver. 1, red. 1 / November 2021 / R. Yulbarisov & A. Mayorov / CRTeam / NRNU MEPhI, Russia

    #   The function generates a particle type according to the chemical composition of galactic cosmic
    #   rays in the Solar system.
    #
    #   Arguments:
    #               E_type      - string            - Type of physical quantity of energy:
    #                                                   'T' for kinetic energy per nucleon [GeV/n]
    #                                                   'E' for total kinetic energy [GeV]
    #                                                   'R' for rigidity [GV]
    #               E           - scalar            - Kinetic energy per nucleon [GeV/n] OR total kinetic energy [GeV]
    #                                                                               OR rigidity [GV] of the particle
    #               PartName    - string            - Empty OR Name of particles according to GetNucleiProp.m
    #               Mod. pot.   - integer           - Empty OR Force-field modulation potential, phi (not Phi), GV
    #
    #   Output:
    #               J           - float array       - Particle's flux at given energy
    #
    #   Examples:
    #               J = GetGCRflux('T', 10, '', '')
    #               J = GetGCRflux('R', 1, 'C-12', '')
    #               J = GetGCRflux('R', 1:1:10, '', '')
    #               J = GetGCRflux('R', 0.5:0.1:1, '', ['C-12','he4','apr','pr','pos','ele'])
    #               J = GetGCRflux('T', 1, '', 0.5)

    # Get particle's list
    PartNameAll = ['proton', 'he4', 'Li-7', 'Be-9', 'B-11', 'C-12', 'N-14', 'O-16', 'F-19', 'Ne-20', 'Na-23',
                   'Mg-24', 'Al-27', 'Si-28', 'Fe-56', 'ele', 'pos', 'apr']

    particle_aliases = {name.replace('-', ''): name for name in PartNameAll if '-' in name}
    particle_aliases.update({
        'p': 'proton',
        'H1': 'proton',
        'H-1': 'proton',
        'He4': 'he4',
        'He-4': 'he4',
        'alpha': 'he4',
        'e-': 'ele',
        'e+': 'pos',
        'anti_proton': 'apr',
        'anti_p': 'apr',
        'p~': 'apr',
        'p_bar': 'apr',
    })

    def _normalize_part_name(name):
        if isinstance(name, str):
            return particle_aliases.get(name, name)
        return [particle_aliases.get(item, item) for item in name]

    if PartName is not None:
        PartName = _normalize_part_name(PartName)
        if isinstance(PartName, str):
            N = PartNameAll.index(PartName)
        else:
            N = [PartNameAll.index(PartName[i]) for i in range(len(PartName))]
    else:
        N = [i for i in range(len(PartNameAll))]

    # Table data

    # Note: Z and A values for e-, e+, and p_bar are set incorrectly to avoid division by 0 and negative
    # rigidity values
    Z = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 26, 1, 1, 1]
    A = [1, 4, 7, 9, 11, 12, 14, 16, 19, 20, 23, 24, 27, 28, 56, 1, 1, 1]
    M = [0.9382720,
         3.7273790,
         6.5338330,
         8.3927490,
         10.252547,
         11.174862,
         13.040202,
         14.895079,
         17.692300,
         18.617728,
         21.409211,
         22.335791,
         25.126499,
         26.053185,
         52.089773,
         0.0005110,
         0.0005110,
         0.9382720]

    J0 = [1872.87554874068,
          139.779118125181,
          0.67360595806831,
          0.33984248416068,
          1.08760559922283,
          3.68261223136148,
          1.01406884838729,
          3.52342241967566,
          0.06742830119460,
          0.53818706959626,
          0.11973060415613,
          0.67412251432422,
          0.12316775376951,
          0.49507921860902,
          0.23975797617632,
          36.5063815538979,
          6.01437512970430,
          0.02021579494618]

    a = [1.61113357145177,
         1.13399984942772,
         1.56511433422331,
         1.52358690395871,
         1.48379096759935,
         1.32184803136140,
         1.47722025525677,
         1.23716432835891,
         1.49951022498618,
         1.51113615332032,
         1.29489790732712,
         1.41840400173818,
         1.42037408858333,
         1.43442256038579,
         1.64447842399986,
         2.14364556649091,
         1.36377570394312,
         6.44997658613249]

    g1 = 0

    g2 = [1.28761628389517,
          1.21715858386835,
          1.50073847406430,
          1.24364031903675,
          1.35693206808129,
          1.28478271904066,
          1.40847267712337,
          1.08737530145770,
          1.05482398425009,
          1.22671674504678,
          0.99296113338822,
          1.09661074771491,
          1.35437363481094,
          0.95735286877508,
          0.83030874570426,
          2.77366285042517,
          6.11040082448321,
          2.79960549756288]

    g3 = [-2.17219702386621,
          -2.21065644455211,
          -2.00837249948625,
          -2.45127118367176,
          -2.22838803240006,
          -2.08425524409385,
          -2.00754622517579,
          -2.46629906200007,
          -2.90115783293288,
          -2.20688620129531,
          -2.98159372048410,
          -2.49504637558846,
          -2.03254549530120,
          -2.87115816939409,
          -3.28332449585784,
          -1.18498108356250,
          -0.49390697550654,
          -1.01848000448823]

    # Calculation of fluxes
    if hasattr(N, '__iter__'):
        T = np.zeros((len(N), len(E)))
    else:
        T = np.zeros(len(E))
    if E_type == 'T':
        s = 0
        if hasattr(N, '__iter__'):
            for i in N:
                T[s] = E
                s += 1
        else:
            T = E
    elif E_type == 'R':
        s = 0
        if hasattr(N, '__iter__'):
            for i in N:
                T[s] = ConvertR2T(E, M[i], A[i], Z[i])
                s += 1
        else:
            T = ConvertR2T(E, M[N], A[N], Z[N])
    elif E_type == 'E':
        s = 0
        if hasattr(N, '__iter__'):
            for i in N:
                T[s] = E / A[i]
                s += 1
        else:
            T = E / A[N]
    else:
        print('GetParticleGCR: Invalid type of physical quantity of energy. ')
        print('Use "T" for kinetic energy per nucleon, "E" for total kinetic energy or "R" for rigidity.')
        exit(-1)

    J_LIS = lambda T, A, M, J0, a, g1, g2, g3: (J0 * np.power(T, g1) *
                                                np.power(((np.power(T, g2) + np.power(a, g2)) / (1 + np.power(a, g2))),
                                                         g3) /
                                                (1 - 1 / np.power((1 + A * T / M), 2)))
    s = 0
    if hasattr(N, '__iter__'):
        J = np.zeros((len(N), len(E)))
        for i in N:
            J[s] = J_LIS(T[s] + Z[i] / A[i] * F, A[i], M[i], J0[i], a[i], g1, g2[i], g3[i]) * T[s] * (
                    T[s] + 2 * 0.93827) / (T[s] + Z[i] / A[i] * F) / (T[s] + Z[i] / A[i] * F + 2 * 0.93827)
            if E_type == 'R':
                dRdT = A[i] / Z[i] / np.sqrt(1 - 1 / np.power((1 + A[i] * T[s] / M[i]), 2))
                J[s] = J[s] / dRdT
            elif E_type == 'E':
                J[s] = J[s] / A[i]
            s += 1
    else:
        J = np.zeros(len(E))
        J = (J_LIS(T + Z[N] / A[N] * F, A[N], M[N], J0[N], a[N], g1, g2[N], g3[N]) * T *
             (T + 2 * 0.93827) / (T + Z[N] / A[N] * F) / (T + Z[N] / A[N] * F + 2 * 0.93827))
        if E_type == 'R':
            dRdT = A[N] / Z[N] / np.sqrt(1 - 1 / np.power((1 + A[N] * T / M[N]), 2))
            J = J / dRdT
        elif E_type == 'E':
            J = J / A[N]

    return J


def _power_law_interp(x, xp, fp):
    # Linear interpolation in the log-log scale, continued as a power law beyond the table
    x, xp, fp = np.log(x), np.log(xp), np.log(fp)
    i = np.clip(np.searchsorted(xp, x), 1, xp.size - 1)
    return np.exp(fp[i - 1] + (x - xp[i - 1]) * (fp[i] - fp[i - 1]) / (xp[i] - xp[i - 1]))


def local_interstellar_spectrum(energy_per_nucleon, particle, model=None):
    """
    Calculate the local interstellar spectrum (LIS) of galactic cosmic rays.

    Parameters
    ----------
    energy_per_nucleon : float or array_like
        Kinetic energy per nucleon in MeV; for electrons and positrons, the
        kinetic energy.
    particle : Particle or str
        Particle object or name. Isotopes share the LIS of their element.
    model : {'Boschini2020', 'Bisschoff2019', 'CustomFit'} or None, optional
        LIS model. By default, 'Bisschoff2019' for electrons and positrons,
        'CustomFit' for antiprotons, and 'Boschini2020' for nuclei.

    Returns
    -------
    float or numpy.ndarray
        Differential intensity in (m2 s sr MeV/n)^-1.

    Notes
    -----
    * 'Boschini2020': numerical tables for H-Ni [1]_ from 1 MeV/n to
      ~100-500 TeV/n, interpolated in the log-log scale.
    * 'Bisschoff2019': analytical expressions for e-, e+, H, He, B, C, and O
      [2]_, given for 3 MeV/n to 100 GeV/n.
    * 'CustomFit': unpublished analytical fit for e-, e+, antiprotons, H-Si, and Fe.

    Outside these energy ranges, the LIS is extrapolated ('Boschini2020' as
    a power law).

    References
    ----------
    .. [1] Boschini, M. J., et al. "Inference of the local interstellar
       spectra of cosmic-ray nuclei Z <= 28 with the GalProp-HelMod
       framework." The Astrophysical Journal Supplement Series 250.2 (2020): 27.
    .. [2] Bisschoff, D., Potgieter, M. S., and Aslam, O. P. M. "New very
       local interstellar spectra for electrons, positrons, protons, and
       light cosmic-ray nuclei." The Astrophysical Journal 878.1 (2019): 59.
    """
    if isinstance(particle, str):
        particle = Particle(Name=particle)
    if particle.A > 0 and particle.Z > 0:
        species = int(particle.Z)
    else:
        species = {11: "e-", -11: "e+", -2212: "anti_proton"}.get(particle.PDG)
    if model is None:
        defaults = {"e-": "Bisschoff2019", "e+": "Bisschoff2019", "anti_proton": "CustomFit"}
        model = defaults.get(species, "Boschini2020")

    T = np.asarray(energy_per_nucleon) / 1e3  # GeV/n
    mass = particle.M / max(particle.A, 1) / 1e3  # GeV per nucleon
    beta2 = 1 - (mass / (T + mass)) ** 2
    match model:
        case "Boschini2020" if species in range(1, 29):
            table_T, table_J = _BOSCHINI2020[_BOSCHINI2020[:, 0] == species, 1:].T
            J = _power_law_interp(T, table_T, table_J)
        case "Bisschoff2019" if species in _BISSCHOFF2019:
            J = _BISSCHOFF2019[species](T, beta2)
        case "CustomFit" if species in _CUSTOM_FIT:
            J0, a, g2, g3 = _CUSTOM_FIT[species]
            J = J0 * ((T ** g2 + a ** g2) / (1 + a ** g2)) ** g3 / beta2
        case _:
            raise ValueError(f"No {model!r} LIS for {particle.Name}")
    return J / 1e3  # (m2 s sr GeV/n)^-1 -> (m2 s sr MeV/n)^-1
