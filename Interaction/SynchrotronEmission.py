'''
11.10.2024
Функция SynchrotronEmission выполняет расчет энергий фотонов, испускаемых заряженной частицей
(T_e_MeV, m, Z) за временной промежуток delta_t, движущейся в магнитном поле B под углом alpha.
Фотоны испускаются вдоль направления движения частицы.
INPUT
    delta_t -   sec     -   временной промежуток, за который частица испускает излучение
    T_e_MeV -   MeV     -   кинетическая энергия частицы
    B       -   Tesla   -   средняя индукция магнитного поля, в котором находится частица за delta_t
    alpha   -   radians -   средний угол между магнитным полем и скоростью частицы за delta_t
    m       -   kg      -   масса частицы
    Z       -   units   -   зарядовое число частицы
OUTPUT
    E_keV_photons   -   keV -  ndarray  -   массив энергий испускаемых фотонов
'''

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid, IntegrationWarning
from scipy.special import kv


def MakeSynchrotronEmission(delta_t, T_MeV, Bsina, M, Z):
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

    # Визуализация распределения энергии фотонов
    if __name__ == '__main__':
        plt.figure()
        plt.hist(E_keV_photons, bins=50, density=True, alpha=0.7)
        plt.xlabel('Photon Energy (keV)')
        plt.ylabel('Probability Density')
        plt.show()

    return E_keV_photons


# Пример вызова функции с исходными данными
if __name__ == '__main__':
    delta_t = 6.9853e-2  # sec
    T_e_MeV = 1e6  # MeV
    B = 1e-5  # T
    alpha = 90  # угол между магнитным полем и скоростью
    sina = np.sin(alpha / 180 * np.pi)
    m_e_MeV = 0.511  # MeV
    Z = -1

    E_keV_photons = MakeSynchrotronEmission(delta_t, T_e_MeV, B*sina, m_e_MeV, Z)


def get_N_avg(B_perp, delta_t, M, Z):
    h = 6.62607015e-34
    eps0 = 8.85418781762039e-12
    e = abs(Z) * 1.602176634e-19
    c = 299792458
    MeV2kg = 1.7826619216224e-30
    m = M * MeV2kg
    return (5 / 2 / np.sqrt(3)) * (e ** 3 * B_perp) / (4 * np.pi * eps0 * h / (2 * np.pi) * c * m) * delta_t