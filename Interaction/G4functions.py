import os
import subprocess
from io import StringIO

import numpy as np
from pymsis import msis
from pyproj import Transformer

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

    :param PDG: Particle PDG code
    :type PDG: int

    :param E: Kinetic energy of the particle [MeV]
    :type E: float

    :param m: Path of a particle in [g/cm^2]
    :type m: float

    :param rho: Density of medium [g/cm^3]
    :type rho: float

    :param w: Medium composition, sum must be equal 1
    :type w: float or array_like

    :return: primary

        - Name - Name

        - PDGcode - PDG encoding

        - Mass - Mass [MeV]

        - Charge - Charge

        - KineticEnergy - Kinetic energy of the particle [MeV]

        - MomentumDirection - Direction of the velocity of the particle [unit vector]

        - Position - Coordinates of the primary particle [m]

        - LastProcess - Name of the last process in which the primary particle participated (usually 'Transportation' or '...Inelastic')

    :rtype: structured ndarray
    """

    # Argument checking
    if np.sum(w) < 0.999:
        raise ValueError('G4Int: total sum of medium fractions is not equal 1')
    # Fractions of H, He, N, O, Ar
    if len(w) != 5:
        raise ValueError('G4Int: wrong number of fractions (atmosphere)')

    # Calling an executable binary program
    path = os.path.dirname(__file__)
    seed = np.random.randint(2147483647)
    cmd = f"'{path}'/MatterLayer {seed} {PDG} {E} {m} {rho} {' '.join(map(str, w))}"
    result = subprocess.run(f"bash -c 'source {path_geant4}/bin/geant4.sh && {cmd}'", shell=True, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode("utf-8"))
        raise RuntimeError('Geant4 program did not work successfully')
    output = result.stdout.decode("utf-8")

    p = output.find('Information about the primary particle')
    s = output.find('Information about the secondary particles')

    # Reading information about the primary particle
    dtype = np.dtype({'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'KineticEnergy', 'MomentumDirection', 'Position', 'LastProcess'],
                      'formats': ['U32', 'i4', 'f8', 'i4', 'f8', '(3,)f8', '(3,)f8', 'U32']})
    primary = np.genfromtxt(StringIO(output[p:s].replace('(', '').replace(')', '')), dtype, delimiter=",", skip_header=2)

    # Reading information about the secondary particles
    secondary = np.array([])
    if s != -1:
        dtype = np.dtype({'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'KineticEnergy', 'MomentumDirection'],
                          'formats': ['U32', 'i4', 'f8', 'i4', 'f8', '(3,)f8']})
        secondary = np.genfromtxt(StringIO(output[s:].replace('(', '').replace(')', '')), dtype, delimiter=",", skip_header=2)
        if secondary.size > 0 and secondary.ndim == 0:
            secondary = np.array([secondary])

    return primary, secondary


def G4Decay(PDG, E):
    """
    The function calls executable binary program that simulate decay of unstable particle and outputs
    information about products.

    :param PDG: Particle PDG code
    :type PDG: int

    :param E: Kinetic energy of the particle [MeV]
    :type E: float

    :return: secondary

        - Name - Name

        - PDGcode - PDG encoding

        - Mass - Mass [MeV]

        - Charge - Charge

        - KineticEnergy - Kinetic energy of the particle [MeV]

        - MomentumDirection - Direction of the velocity of the particle [unit vector]
    :rtype: structured ndarray

    Examples
        ``secondary = G4Decay(2112, 1)        # n -> p + e- + anti_nu_e``

        ``secondary = G4Decay(-2112, 1)       # anti_n -> anti_p + e+ + nu_e``

        ``secondary = G4Decay(211, 1)         # pi+ -> mu+ + nu_mu``

        ``secondary = G4Decay(13, 1)          # mu- -> e- + anti_nu_e + nu_mu``

        ``secondary = G4Decay(1000060140, 1)  # C14 -> N14 + e- + anti_nu_e``

        ``secondary = G4Decay(1000922380, 1)  # U238 -> Th234 + alpha``

        ``secondary = G4Decay(2212, 1)        # p is stable``
    """

    # Calling an executable binary program
    path = os.path.dirname(__file__)
    seed = np.random.randint(2147483647)
    cmd = f"'{path}'/DecayGenerator {seed} {PDG} {E}"
    result = subprocess.run(f"bash -c 'source {path_geant4}/bin/geant4.sh && {cmd}'", shell=True, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode("utf-8"))
        raise RuntimeError('Geant4 program did not work successfully')
    output = result.stdout.decode("utf-8")

    s = output.find('Information about the secondary particles')

    # Reading information about the secondary particles
    secondary = np.array([])
    if s != -1:
        dtype = np.dtype({'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'LifeTime', 'KineticEnergy', 'MomentumDirection'],
                          'formats': ['U32', 'i4', 'f8', 'i4', 'f8', 'f8', '(3,)f8']})
        secondary = np.genfromtxt(StringIO(output[s:].replace('(', '').replace(')', '')), dtype, delimiter=",", skip_header=2)

    return secondary


def G4Shower(PDG, E, r, v, date):
    """
    The function calls executable binary program that calculates interaction of the charged particle
    with atmospheric column of height h and outputs information about secondary (albedo) particles.

    The program creates a cylindrical column of air, which is divided into layers layers of 1 km thick.
    The air density for each layer is assumed to be constant and is calculated from the
    atmospheric model NRLMSISE-00. When a particle enters the atmosphere at an angle, the horizontal
    velocity component is directed along the X axis in the coordinate system of the cylinder.

    :param PDG: Particle PDG code
    :type PDG: int

    :param E: Kinetic energy of the particle [MeV]
    :type E: float

    :param r: Coordinates of the primary particle in GEO [m]
    :type r: float array

    :param v: Velocity of the primary particle in GEO [unit vector]
    :type v: float array

    :param date: Current datetime
    :type date: datetime

    :return: primary

            - Name - Name

            - PDGcode - PDG encoding

            - Mass - Mass [MeV]

            - Charge - Charge

            - PositionInteraction - Coordinates of the interaction of the primary particle [m]

            - LastProcess - Name of the last process in which the primary particle participated

            secondary

            - Name - Name

            - PDGcode - PDG encoding

            - Mass - Mass [MeV]

            - Charge - Charge

            - KineticEnergy - Kinetic energy of the particle [MeV]

            - MomentumDirection - Direction of the velocity of the particle in GEO [unit vector]

            - Position - Coordinates of the secondary (albedo) particle in GEO [m]

    :rtype primary: structured ndarray
    :rtype secondary: structured ndarray

    Examples:
        ``primary, secondary = G4Shower(2212, 10e3, 95, 10, 1, 0, 0, 0, 150, 150, 4)``

        ``primary, secondary = G4Shower(1000020040, 20e3, 80, 30, 365, 43200, -80, 270, 200, 210, 80)``
    """

    # Calculating input parameters
    geo_to_lla = Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                                      {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'})
    lon, lat, alt = geo_to_lla.transform(r[0], r[1], r[2], radians=False)
    earth_radius = (np.linalg.norm(r) - alt) / 1e3 # m -> km
    # day of year (from 1 to 365 or 366)
    doy = date.timetuple().tm_yday
    # seconds in day
    sec = (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    f107, f107a, ap = msis.get_f107_ap(date)

    # Calling an executable binary program
    path = os.path.dirname(__file__)
    seed = np.random.randint(2147483647)
    cmd = f"'{path}'/Atmosphere {seed} {PDG} {E} {r[0] / 1e3} {r[1] / 1e3} {r[2] / 1e3} {v[0]} {v[1]} {v[2]} {earth_radius} {doy} {sec} {lat} {lon} {f107a} {f107} {ap[0]}"
    result = subprocess.run(f"bash -c 'source {path_geant4}/bin/geant4.sh && {cmd}'", shell=True, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode("utf-8"))
        raise RuntimeError('Geant4 program did not work successfully')
    output = result.stdout.decode("utf-8")

    p = output.find('Information about the primary particle')
    s = output.find('Information about the secondary particles')

    # Reading information about the primary particle
    dtype = np.dtype({'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'PositionInteraction', 'LastProcess'],
                      'formats': ['U32', 'i4', 'f8', 'i4', '(3,)f8', 'U32']})
    primary = np.genfromtxt(StringIO(output[p:s].replace('(', '').replace(')', '')), dtype, delimiter=",", skip_header=2)

    # Reading information about the secondary particles
    secondary = np.array([])
    if s != -1:
        dtype = np.dtype({'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'KineticEnergy', 'MomentumDirection', 'Position'],
                          'formats': ['U32', 'i4', 'f8', 'i4', 'f8', '(3,)f8', '(3,)f8']})
        secondary = np.genfromtxt(StringIO(output[s:].replace('(', '').replace(')', '')), dtype, delimiter=",", skip_header=2)
        if secondary.size > 0 and secondary.ndim == 0:
            secondary = np.array([secondary])

    return primary, secondary
