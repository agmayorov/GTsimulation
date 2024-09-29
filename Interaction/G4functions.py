import subprocess
import os
import io
import numpy as np
from .settings import path_geant4


def G4Interaction(PDG, E, m, rho, w):
    """
    The function calls executable binary program that calculate interaction of the charge particle
    with matter at a given path length and outputs information about secondary particles.

    For this we simulate a cylinder filled with matter with a density rho. Cylinder length is calculated
    as l = m / rho. The radius of the cylinder R is equal to its length l. The initial coordinate of
    the particle is (0, 0, 0). The initial velocity is directed along the cylinder axis, which coincides
    with the Z axis. The simulation stops when the primary particle has died or reached the boundary
    of the cylinder.

    Parameters:
        PDG         - int                   - Particle PDG code
        E           - float                 - Kinetic energy of the particle [MeV]
        m           - float                 - Path of a particle in [g/cm^2]
        rho         - float                 - Density of medium [g/cm^3]
        w           - array_like of float   - Medium composition, sum must be equal 1

    Returns:
        primary     - structured ndarray
                        Name                - Name
                        PDGcode             - PDG encoding
                        Mass                - Mass [MeV]
                        Charge              - Charge
                        KineticEnergy       - Kinetic energy of the particle [MeV]
                        MomentumDirection   - Direction of the velocity of the particle (unit vector)
                        Position            - Coordinates of the primary particle [m]
                        LastProcess         - Name of the last process in which the primary particle
                                              participated (usually 'Transportation' or '...Inelastic')
        secondary   - structured ndarray
                        Name, PDGcode, Mass, KineticEnergy, MomentumDirection
    """

    # Argument checking
    if np.sum(w) < 0.999:
        raise ValueError('G4Int: total sum of medium fractions is not equal 1')
    # Fractions of H, He, N, O, Ar
    if len(w) != 5:
        raise ValueError('G4Int: wrong number of fractions (atmosphere)')
    # TEMPORARY PATCH
    if E == 0:
        E = 1

    # Calling an executable binary program
    path = os.path.dirname(__file__)
    result = subprocess.run(f"bash {path_geant4}/bin/geant4.sh; {path}/MatterLayer "
                            f"{PDG} {E} {m} {rho} {' '.join(map(str, w))}", shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError('Geant4 program did not work successfully')
    output = result.stdout.decode("utf-8")

    p = output.find('Information about the primary particle')
    s = output.find('Information about the secondary particles')

    # Reading information about the primary particle
    dtype = np.dtype({'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'KineticEnergy', 'MomentumDirection', 'Position', 'LastProcess'],
                      'formats': ['U32', 'i4', 'f8', 'i4', 'f8', '(3,)f8', '(3,)f8', 'U32']})
    primary = np.genfromtxt(io.StringIO(output[p:s].replace('(', '').replace(')', '')), dtype, delimiter=",", skip_header=2)

    # Reading information about the secondary particles
    secondary = np.array([])
    if s != -1:
        dtype = np.dtype({'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'KineticEnergy', 'MomentumDirection'],
                          'formats': ['U32', 'i4', 'f8', 'i4', 'f8', '(3,)f8']})
        secondary = np.genfromtxt(io.StringIO(output[s:].replace('(', '').replace(')', '')), dtype, delimiter=",", skip_header=2)

    return primary, secondary


def G4Decay(PDG, E):
    """
    The function calls executable binary program that simulate decay of unstable particle and outputs
    information about products.
 
    Parameters:
        PDG         - int                   - Particle PDG code
        E           - float                 - Kinetic energy of the particle [MeV]

    Returns:
        secondary   - structured ndarray
                        Name                - Name
                        PDGcode             - PDG encoding
                        Mass                - Mass [MeV]
                        Charge              - Charge
                        KineticEnergy       - Kinetic energy of the particle [MeV]
                        MomentumDirection   - Direction of the velocity of the particle (unit vector)

    Examples:
        secondary = G4Decay(2112, 1)        # n -> p + e- + anti_nu_e
        secondary = G4Decay(-2112, 1)       # anti_n -> anti_p + e+ + nu_e
        secondary = G4Decay(211, 1)         # pi+ -> mu+ + nu_mu
        secondary = G4Decay(13, 1)          # mu- -> e- + anti_nu_e + nu_mu
        secondary = G4Decay(1000060140, 1)  # C14 -> N14 + e- + anti_nu_e
        secondary = G4Decay(1000922380, 1)  # U238 -> Th234 + alpha
        secondary = G4Decay(2212, 1)        # p is stable
    """

    # Calling an executable binary program
    path = os.path.dirname(__file__)
    result = subprocess.run(f"bash {path_geant4}/bin/geant4.sh; {path}/DecayGenerator "
                            f"{PDG} {E}", shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError('Geant4 program did not work successfully')
    output = result.stdout.decode("utf-8")

    s = output.find('Information about the secondary particles')

    # Reading information about the secondary particles
    secondary = np.array([])
    if s != -1:
        dtype = np.dtype({'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'LifeTime', 'KineticEnergy', 'MomentumDirection'],
                          'formats': ['U32', 'i4', 'f8', 'i4', 'f8', 'f8', '(3,)f8']})
        secondary = np.genfromtxt(io.StringIO(output[s:].replace('(', '').replace(')', '')), dtype, delimiter=",", skip_header=2)

    return secondary


# def G4Shower(PDG, E, h, alpha, doy, sec, lat, lon, f107A, f107, ap):
#     res = subprocess.run(f". /lustre/incos/set_pam_env.sh 11 4.11.00.p02; "
#                          f"/lustre/mFunctions/G4Interaction/AtmosphericColumn/build/AtmosphericColumn {PDG} {E} {h} "
#                          f"{alpha} {doy} {sec} {lat} {lon} {f107A} {f107} {ap}", shell=True, stdout=subprocess.PIPE)
#     if res.returncode != 0:
#         print(res.stdout)
#         raise Exception("Geant4 program did not work successfully")

#     output = res.stdout
#     output = output.decode("utf-8")

#     k = output.find('Information about the primary particle:')
#     if k == -1:
#         raise RuntimeError('Primary particle information not found')

#     segment = output[k:]

#     name_match = re.search(r'Particle name: (.+)', segment)
#     position_match = re.search(r'Position of interaction: \((.+?),(.+?),(.+?)\)', segment)
#     process_name_match = re.search(r'Process name: (.+)', segment)

#     if not (position_match and name_match and process_name_match):
#         raise RuntimeError('Primary particle information format incorrect')

#     r_int = np.array([float(position_match.group(1)), float(position_match.group(2)),
#                       float(position_match.group(3))]) / 1e6  # [mm] to [km]
#     ParticleName = name_match.group(1).strip()
#     process = process_name_match.group(1).strip()

#     albedo = []

#     k = [m.start() for m in re.finditer('Albedo particles:', output)]
#     if k:
#         segment_albedo = output[k[0] + 18:]
#         for seg in segment_albedo.split("\n")[:-1]:
#             params = seg.split()
#             particle_name = params[0]
#             radius = [float(n) / 1e6 for n in params[1][1:-1].split(",")]
#             momentum_direction = [float(n) for n in params[2][1:-1].split(",")]
#             kinetic_energy = float(params[3]) / 1e3
#             pdg, mass, charge = float(params[4]), float(params[5]), float(params[6])

#             albedo.append({
#                 'ParticleName': particle_name,
#                 'r': radius,
#                 'v': momentum_direction,
#                 'E': kinetic_energy,
#                 'PDG': [pdg, mass, charge]
#             })

#     return ParticleName, r_int, process, albedo
