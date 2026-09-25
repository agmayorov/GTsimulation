import os
import sys
import warnings

import numpy as np
from scipy.integrate import quad, cumulative_trapezoid, IntegrationWarning
from scipy.special import kv


def MakeSynchrotronEmission(delta_t, T_MeV, Bsina, M, Z):
    """
    Calculate the energies of synchrotron-emitted photons.

    Parameters
    ----------
    delta_t : float
        Time interval during which the particle emits radiation, s.
    T_MeV : float
        Kinetic energy of the particle, MeV.
    Bsina : float
        Perpendicular component of the magnetic field, T.
    M : float
        Particle mass in MeV/c².
    Z : int or float
        Charge number of the particle.

    Returns
    -------
    E_keV_photons : numpy.ndarray
        Array of emitted photon energies, keV.

    Notes
    -----
    The photons are assumed to be emitted along the direction of particle motion.
    """
    # Константы
    h = 6.62607015e-34  # kg*m^2/sec
    eps0 = 8.85418781762039e-12  # m^-3*kg^-1*sec^4*A^2
    e = abs(Z) * 1.602176634e-19  # A*sec (Coulomb)
    c = 299792458  # m/sec
    MeV2kg = 1.7826619216224e-30  # MeV/c^2 to kg conversion
    J2keV = (1e-3 * MeV2kg * c ** 2) ** -1

    # Параметры
    m = M * MeV2kg  # kg
    T_e = T_MeV * MeV2kg * c ** 2  # kg*m^2/sec^2
    G = T_e / m / c ** 2 + 1  # Лоренц-фактор
    wc = 3 / 2 * G ** 2 * e * Bsina / m  # rad/sec, критическая частота фотонов
    Ec_keV = wc * h / (2 * np.pi) * J2keV

    # Число испускаемых фотонов
    N_avg = (5 / 2 / np.sqrt(3)) * (e ** 3 * Bsina) / (4 * np.pi * eps0 * h / (2 * np.pi) * c * m) * delta_t

    # Энергия фотонов
    E_keV_vec = np.logspace(np.log10(0.000000005 * Ec_keV), np.log10(10 * Ec_keV), 1000)

    # Функции для интегралов и вычислений
    def E(E_keV):
        return E_keV * 1e-3 * MeV2kg * c ** 2  # kg*m^2/sec^2

    def x(E_keV):
        return E(E_keV) * 4 * np.pi * m / (3 * G ** 2 * e * h * Bsina)

    def F(E_keV):
        # Подавление IntegrationWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)

            # Используем менеджер контекста для безопасного перенаправления stderr
            with open(os.devnull, 'w') as f:
                sys.stderr = f  # Перенаправляем stderr на devnull

                x_values = x(E_keV)
                result = x_values * np.array([quad(lambda z: kv(5 / 3, z), y, np.inf)[0] for y in x_values])

            # Восстанавливаем stderr
            sys.stderr = sys.__stderr__

            return result
    
    # Нормализованная функция
    def S(E_keV):
        return 9 * np.sqrt(3) / (8 * np.pi) * F(E_keV)

    S_vec = S(E_keV_vec)

    # Вычисление CDF
    cdf_values = cumulative_trapezoid(S_vec / E_keV_vec, E_keV_vec, initial=0)
    cdf_values /= max(cdf_values)

    # Энергии испускаемых фотонов
    E_keV_photons = np.interp(np.random.rand(int(N_avg)), cdf_values, E_keV_vec)

    return E_keV_photons


def get_N_avg(B_perp, delta_t, M, Z):
    """
    Calculate the average number of emitted synchrotron photons.

    Parameters
    ----------
    B_perp : float
        Perpendicular component of the magnetic field, T.
    delta_t : float
        Time interval, s.
    M : float
        Particle mass in MeV/c².
    Z : int or float
        Charge number of the particle.

    Returns
    -------
    float
        Average number of emitted photons.
    """
    h = 6.62607015e-34
    eps0 = 8.85418781762039e-12
    e = abs(Z) * 1.602176634e-19
    c = 299792458
    MeV2kg = 1.7826619216224e-30
    m = M * MeV2kg
    return (5 / 2 / np.sqrt(3)) * (e ** 3 * B_perp) / (4 * np.pi * eps0 * h / (2 * np.pi) * c * m) * delta_t
