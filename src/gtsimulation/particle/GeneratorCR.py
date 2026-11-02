from pathlib import Path

import numpy as np

from gtsimulation.particle._particle import Particle

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


def gcr_flux(energy_per_nucleon, particle, modulation_potential, model=None):
    """
    Calculate the flux of galactic cosmic rays in the force-field approximation [1]_.

    Parameters
    ----------
    energy_per_nucleon : float or array_like
        Kinetic energy per nucleon in MeV; for electrons and positrons, the
        kinetic energy.
    particle : Particle or str
        Particle object or name.
    modulation_potential : float
        Modulation potential in MV.
    model : str or None, optional
        LIS model, see `local_interstellar_spectrum`.

    Returns
    -------
    float or numpy.ndarray
        Differential intensity in (m2 s sr MeV/n)^-1.

    References
    ----------
    .. [1] Gleeson, L. J., and Axford, W. I. "Solar modulation of galactic
       cosmic rays." The Astrophysical Journal 154 (1968): 1011.
    """
    if isinstance(particle, str):
        particle = Particle(Name=particle)
    T = np.asarray(energy_per_nucleon)
    mass = particle.M / max(particle.A, 1)  # MeV per nucleon
    T_lis = T + abs(float(particle.Z)) / max(particle.A, 1) * modulation_potential  # Particle.Z is a Fraction
    return local_interstellar_spectrum(T_lis, particle, model) * T * (T + 2 * mass) / (T_lis * (T_lis + 2 * mass))


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
