from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator


_CDF_INTERPOLATOR = None


def initialize():
    """
    Initialize the pair-production module.

    Loads the energy-distribution interpolation table on the first call.
    Subsequent calls reuse the already initialized interpolator.
    """
    _get_cdf_interpolator()


def _get_cdf_interpolator():
    global _CDF_INTERPOLATOR

    if _CDF_INTERPOLATOR is None:
        data_path = Path(__file__).resolve().with_name(
            "epsilon_interpolation_data.npz"
        )

        with np.load(data_path) as data:
            epsilon_table = data["epsilon_table"]
            chi_grid = data["chi_grid"]
            prob_grid = data["prob_grid"]

        _CDF_INTERPOLATOR = RegularGridInterpolator(
            points=(np.log10(chi_grid), prob_grid),
            values=epsilon_table,
            method="cubic",
            bounds_error=False,
            fill_value=0.5
        )

    return _CDF_INTERPOLATOR


@njit(fastmath=True)
def _pair_conversion_probability(T, V, B_tesla, path):
    """
    Calculate the magnetic pair-production probability for a photon.

    Parameters
    ----------
    T : float
        Photon energy, MeV.
    V : numpy.ndarray
        Photon propagation direction.
    B_tesla : numpy.ndarray
        Magnetic field, T.
    path : float
        Photon path length during the current step, m.

    Returns
    -------
    chi : float
        Quantum nonlinearity parameter.
    chance : float
        Pair-production probability over the given path.
    """
    MASS_E = 0.5109989461        # MeV/c^2
    B_CRIT = 4.414e13            # G
    ALPHA = 7.2973525664e-3
    LAMBDA_C = 3.8615926744e-13  # m

    B_gauss = B_tesla * 1e4
    B_abs = np.linalg.norm(B_gauss)

    if B_abs == 0.0:
        return 0.0, 0.0

    V_abs = np.linalg.norm(V)

    if V_abs == 0.0:
        return 0.0, 0.0

    cos_theta = np.dot(V, B_gauss) / (V_abs * B_abs)

    sin_theta = np.sqrt(1.0 - cos_theta ** 2)

    if sin_theta < 1e-15:
        return 0.0, 0.0

    threshold_energy = 2.0 * MASS_E / sin_theta

    if T < threshold_energy:
        return 0.0, 0.0

    chi = (
        T / (2.0 * MASS_E)
        * (B_abs / B_CRIT)
        * sin_theta
    )

    if chi < 0.1:
        R = (
            0.23
            * (ALPHA / LAMBDA_C)
            * (B_abs * sin_theta) / B_CRIT
            * np.exp(-4.0 / (3.0 * chi))
        )

    elif chi > 10.0:
        R = (
            0.38
            * (ALPHA / LAMBDA_C)
            * (B_abs * sin_theta) / B_CRIT
            * chi ** (-1.0 / 3.0)
        )

    else:
        chi23 = chi ** (2.0 / 3.0)

        f = (
            1.0 + 0.5218 * chi23
        ) / (
            1.0
            + 0.8526
            * (
                chi23
                + 0.1632 * chi ** (4.0 / 3.0)
            )
        )

        R = (
            0.46
            * (ALPHA / LAMBDA_C)
            * (B_abs * sin_theta) / B_CRIT
            * chi ** (-1.0 / 3.0)
            * np.exp(
                (-4.0 / (3.0 * chi)) * f
            )
        )

    chance = 1.0 - np.exp(-R * path)

    return chi, chance


def _produce_pair_energy(T, chi):
    """
    Sample the energy sharing between the electron and positron.

    Parameters
    ----------
    T : float
        Photon energy, MeV.
    chi : float
        Quantum nonlinearity parameter.

    Returns
    -------
    E1 : float
        Total energy of the first particle, MeV.
    E2 : float
        Total energy of the second particle, MeV.
    epsilon : float
        Fraction of the photon energy carried by the first particle.
    """
    MASS_E = 0.5109989461  # MeV/c^2

    e_min = MASS_E / T
    e_max = 1.0 - e_min

    rnd = np.random.random()

    is_mirrored = rnd > 0.5

    u_normalized = (
        rnd if not is_mirrored else 1.0 - rnd
    ) * 2.0

    interpolator = _get_cdf_interpolator()

    one_side_eps = interpolator(
        [np.log10(chi), u_normalized]
    )[0]

    epsilon = (
        one_side_eps
        if not is_mirrored
        else 1.0 - one_side_eps
    )

    if epsilon < e_min or epsilon > e_max:
        raise ValueError(
            "The pair energy fraction is below the rest-energy threshold!"
        )

    E1 = T * epsilon
    E2 = T - E1

    return E1, E2, epsilon


@njit(fastmath=True)
def _sample_pair_momenta(T, V, B, epsilon):
    """
    Sample electron and positron momentum directions.

    Parameters
    ----------
    T : float
        Photon energy, MeV.
    V : numpy.ndarray
        Photon propagation direction.
    B : numpy.ndarray
        Magnetic field, G.
    epsilon : float
        Energy fraction carried by the first particle.

    Returns
    -------
    P1_lab, P2_lab : numpy.ndarray
        Electron and positron momenta in the laboratory frame.
    """
    MASS_E_C2 = 0.5109989461  # MeV/c^2

    b_axis = B / np.linalg.norm(B)

    cos_theta = np.dot(V, b_axis) / np.linalg.norm(V)

    sin_theta = np.sqrt(1.0 - cos_theta ** 2)

    omega_prime = T * sin_theta

    E1_prime = epsilon * omega_prime
    E2_prime = (1.0 - epsilon) * omega_prime

    # Simplest approximation:
    # longitudinal momentum in the perpendicular frame is zero.
    p_z_prime = 0.0

    p_perp1_prime = np.sqrt(
        np.maximum(
            0.0,
            E1_prime ** 2
            - p_z_prime ** 2
            - MASS_E_C2 ** 2
        )
    )

    p_perp2_prime = np.sqrt(
        np.maximum(
            0.0,
            E2_prime ** 2
            - p_z_prime ** 2
            - MASS_E_C2 ** 2
        )
    )

    x_axis = V - cos_theta * b_axis
    norm_x = np.linalg.norm(x_axis)

    if norm_x > 1e-10:
        x_axis /= norm_x

    else:
        if abs(b_axis[0]) < 0.8:
            x_axis = (
                np.array([1.0, 0.0, 0.0])
                - b_axis[0] * b_axis
            )
        else:
            x_axis = (
                np.array([0.0, 1.0, 0.0])
                - b_axis[1] * b_axis
            )

        x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(b_axis, x_axis)

    p1_local_prime = np.array(
        [p_perp1_prime, 0.0, p_z_prime]
    )

    p2_local_prime = np.array(
        [p_perp2_prime, 0.0, -p_z_prime]
    )

    beta = cos_theta
    gamma = (
        1.0 / sin_theta
        if sin_theta > 1e-10
        else 1.0
    )

    def lorentz_boost(
        E_prime,
        p_prime_vec,
        beta,
        gamma,
        x_dir,
        y_dir,
        z_dir
    ):
        p_x_prime = p_prime_vec[0]
        p_y_prime = p_prime_vec[1]
        p_z_prime = p_prime_vec[2]

        p_z_lab = gamma * (
            p_z_prime + beta * E_prime
        )

        p_x_lab = p_x_prime
        p_y_lab = p_y_prime

        p_lab_vector = (
            p_x_lab * x_dir
            + p_y_lab * y_dir
            + p_z_lab * z_dir
        )

        return p_lab_vector

    P1_lab = lorentz_boost(
        E1_prime,
        p1_local_prime,
        beta,
        gamma,
        x_axis,
        y_axis,
        b_axis
    )

    P2_lab = lorentz_boost(
        E2_prime,
        p2_local_prime,
        beta,
        gamma,
        x_axis,
        y_axis,
        b_axis
    )

    return P1_lab, P2_lab


def produce_pair(T, V, B_tesla, path):
    """
    Simulate magnetic photon conversion into an electron-positron pair.

    Parameters
    ----------
    T : float
        Photon energy, MeV.
    V : numpy.ndarray
        Photon propagation direction.
    B_tesla : numpy.ndarray
        Magnetic field, T.
    path : float
        Photon path length during the current step, m.

    Returns
    -------
    result : tuple or None
        ``None`` if no conversion occurs.

        Otherwise returns::

            (
                (
                    (11, T_electron, V_electron),
                    (-11, T_positron, V_positron)
                ),
                epsilon
            )

        where ``epsilon`` is the sampled energy fraction.
    """
    MASS_E = 0.5109989461  # MeV/c^2

    chi, chance = _pair_conversion_probability(
        T,
        V,
        B_tesla,
        path
    )

    if chance <= 0.0:
        return None

    if np.random.random() >= chance:
        return None

    E1, E2, epsilon = _produce_pair_energy(
        T,
        chi
    )

    T1 = E1 - MASS_E
    T2 = E2 - MASS_E

    B_gauss = B_tesla * 1e4

    p1, p2 = _sample_pair_momenta(
        T,
        V,
        B_gauss,
        epsilon
    )

    v1 = p1 / np.linalg.norm(p1)
    v2 = p2 / np.linalg.norm(p2)

    pair = (
        (11, T1, v1),
        (-11, T2, v2)
    )

    return pair, epsilon