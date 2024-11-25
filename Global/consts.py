from enum import Enum
import numpy as np

class Constants:
    c = 2.99792458e8  # Speed of light in m/s
    e = 1.602176634e-19  # Elementary charge in C

    hbar = 6.582119514e-22  # MeV*s
    hbarc = 197.3269788  # MeV*fm

    ao = 0.5291772086e-8  # Bohr radius of hydrogen, cm

    alpha = 7.2973525664e-3  # Fine-structure constant

    u = 4.00260325413  # Unified atomic mass unit to MeV/c^2
    me = 0.5109989461  # Electron mass, MeV/c^2
    mhe4 = 3728.40129745092  # Alpha-particle mass, MeV/c^2


class Units:
    MeV2kg = 1.7826619216224e-30  # MeV/c2 to kg conversion
    MeV2g = 1.7826619216224e-27  # MeV/c2 to  g conversion

    km2m = 1e3

    AU2m = 149.597870700e9
    AU2km = 149.597870700e6

    pc2m = 3.08567758149e16
    kpc2m = 3.08567758149e19

    fm2cm = 1e-13

    RE2m = 6378137.1
    RE2km = 6378.1371
    RM2m = 1737400
    RM2km = 1737.4

    T2nT = 1e9

    # System of units
    # Length
    meters = 1
    kilometers = 1e3 * meters
    earthradius = 6378137.1 * meters
    parsec = 3.08567758149e16 * meters
    kiloparsec = 1e3 * parsec
    astronomicalunits = 149.597870700e9 * meters

    # Symbols
    m = meters
    km = kilometers
    RE = earthradius
    pc = parsec
    kpc = kiloparsec
    AU = astronomicalunits

    # Energy
    megaelectronvolt = 1
    electronvolt = 1e-6 * megaelectronvolt
    kiloelectronvolt = 1e-3 * megaelectronvolt
    gigaelectronvolt = 1e3 * megaelectronvolt
    teraelectronvolt = 1e6 * megaelectronvolt
    petaelectronvolt = 1e9 * megaelectronvolt

    # Symbols
    eV = electronvolt
    keV = kiloelectronvolt
    MeV = megaelectronvolt
    GeV = gigaelectronvolt
    TeV = teraelectronvolt
    PeV = petaelectronvolt

    # Angles
    degree = 1
    radian = 180 / np.pi

    # Symbols
    deg = degree
    rad = radian


class Origins(Enum):
    Galactic = 1
    Albedo = 2
    QuasiTrapped = 3
    Precipitated = 4
    Trapped = 5
    Unknown = 6
