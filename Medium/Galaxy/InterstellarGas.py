import numpy as np
from scipy.optimize import root_scalar
from numpy.polynomial.polynomial import polyval
from numba import jit

from Global import Regions, Units
from Medium import GTGeneralMedium


def fzero(func, x0):
    result = root_scalar(func, bracket=[x0 - 1, x0 + 1], method='brentq')
    return result.root if result.converged else -np.inf


default_params = {
    'X_CO': 2e20,

    # Radial profile for disk and arms
    'r_s': 8.0,
    'r_0': 1.27,
    'r_h': 6.34,
    'h_i': 6.38,
    'f_d_HI': lambda R: 0.2291 * R ** 2 * np.exp(-R / 3.016) / 0.250,  # normalized on n8
    'n8_d_HI': 0.160,
    'n8_d_CO': 0.894,

    # Warped disk
    'z_s_HI': 0.0942,
    'z_s_CO': 0.103,
    'r_z0': 8.5,
    'r_z': 6.94,
    'w_0p': np.array([2.063e-4, -0.002329, 0.001374, -0.03613]),  # NEED TO FIT
    'w_1p': np.array([-6.838e-5, -0.005807, 0.063540, -0.11340]),  # NEED TO FIT
    'w_2p': np.array([1.217e-4, 0.001202, -0.058400, 0.26010]),  # NEED TO FIT
    'theta_1': 4.61,
    'theta_2': 2.73,

    # Central bulge
    'n_b': 47.8,
    'theta_b': 5.67,
    'x_0': 0.751,
    'r_b': 0.514,
    'z_b': 6.43,
    'e_i': 0.647,
    'p_i': 1.18,

    # Spiral arms
    'a': np.array([3.30, 4.35, 5.32, 4.75]),
    'r_min': np.array([2.00, 3.31, 3.89, 3.19]),
    'theta_min': -np.pi / 6 + np.array([1, 2, 3, 4]) * np.pi / 2,
    'sigma_arms': 0.6,
    'n8_s_HI': np.array([0.184, 0.193, 0.332, 0.521]),
    'n8_s_CO': np.array([0.642, 0, 3.37, 7.53])
}


class InterstellarGas(GTGeneralMedium):
    def __init__(self, **kwargs):
        super().__init__()
        self.region = Regions.Magnetosphere
        self.model = "NRLMSIS"

        params = default_params.copy()
        params.update(kwargs)

        self.p = params

    def calculate_model(self, X, Y, Z, *args, **kwargs):
        # Extract individual parameters to pass to the numba-accelerated function
        # self.n_HI, self.n_H2 = self._calculate_model_static(
        #     X, Y, Z,
        #     self.p['X_CO'], self.p['r_s'], self.p['r_0'], self.p['r_h'], self.p['h_i'],
        #     self.p['z_s_HI'], self.p['z_s_CO'], self.p['r_z0'], self.p['r_z'],
        #     np.array(self.p['w_0p']), np.array(self.p['w_1p']), np.array(self.p['w_2p']),
        #     self.p['theta_1'], self.p['theta_2'], self.p['n8_d_HI'], self.p['n8_d_CO'],
        #     self.p['n_b'], self.p['theta_b'], self.p['x_0'], self.p['r_b'], self.p['z_b'],
        #     self.p['e_i'], self.p['p_i'], np.array(self.p['a']), np.array(self.p['r_min']),
        #     np.array(self.p['theta_min']), self.p['sigma_arms'],
        #     np.array(self.p['n8_s_HI']), np.array(self.p['n8_s_CO'])
        # )

        X, Y, Z = X/Units.kpc2m, Y/Units.kpc2m, Z/Units.kpc2m

        # Convert cartesian to polar coordinates
        theta, R, _ = np.arctan2(Y, X), np.hypot(X, Y), Z

        # --- Warped disk ---
        # Radial profile
        f_d_HI = self.p['f_d_HI'](R)
        f_d_CO = np.exp(-(R - self.p['r_s']) / self.p['r_0']) * (1 - np.exp(-(R / self.p['r_h']) ** self.p['h_i']))

        # Warp of disk
        w_0 = polyval(R, self.p['w_0p'][::-1])
        w_1 = polyval(R, self.p['w_1p'][::-1])
        w_2 = polyval(R, self.p['w_2p'][::-1])
        z_0 = w_0 + w_1 * np.sin(theta - self.p['theta_1']) + w_2 * np.sin(2 * theta - self.p['theta_2'])

        # Vertical profile
        ZZ = Z - z_0
        z_h_HI = self.p['z_s_HI'] * np.exp((R - self.p['r_z0']) / self.p['r_z'])
        z_h_CO = self.p['z_s_CO'] * np.exp((R - self.p['r_z0']) / self.p['r_z'])
        f_s_HI = 1/np.cosh(ZZ / z_h_HI) ** 0.5
        f_s_CO = 1/np.cosh(ZZ / z_h_CO) ** 2.0

        # Final disk
        n_d_HI = self.p['n8_d_HI'] * f_d_HI * f_s_HI
        n_d_CO = self.p['n8_d_CO'] * f_d_CO * f_s_CO

        # --- Central bulge ---
        XX = X * np.cos(self.p['theta_b']) + Y * np.sin(self.p['theta_b']) + self.p['x_0']
        YY = -X * np.sin(self.p['theta_b']) + Y * np.cos(self.p['theta_b'])
        RR = np.sqrt(XX ** 2 + (YY / 0.3) ** 2)
        Rr = 1 / (RR / self.p['r_b'] + Z / self.p['z_b'])
        n_b = self.p['n_b'] * np.exp(-Rr ** self.p['e_i']) * Rr ** self.p['p_i']

        # --- Spiral arms ---
        r_distance = np.zeros(4)
        for i in range(4):
            # Find the closest point
            fun = lambda x: self.p['r_min'][i] / R * np.exp((x - self.p['theta_min'][i]) / self.p['a'][i]) - np.cos(
                x - theta) + self.p['a'][i] * np.sin(x - theta)
            theta_guess = np.linspace(self.p['theta_min'][i],
                                      self.p['a'][i] * np.log(25 / self.p['r_min'][i]) + self.p['theta_min'][i], 100)
            zero_crossing = fun(theta_guess) * np.roll(fun(theta_guess), 1) <= 0
            zero_crossing[[0, -1]] = False
            theta_guess = theta_guess[zero_crossing]
            if theta_guess.size == 0:
                continue
            theta_candidate = np.array([fzero(fun, tg) for tg in theta_guess])
            r_candidate = self.p['r_min'][i] * np.exp((theta_candidate - self.p['theta_min'][i]) / self.p['a'][i])
            X_candidate, Y_candidate = r_candidate * np.cos(theta_candidate), r_candidate * np.sin(theta_candidate)
            r_distance[i] = np.min(np.sqrt((X_candidate - X) ** 2 + (Y_candidate - Y) ** 2))

        n_s_HI = np.sum(
            self.p['n8_s_HI'] * f_d_HI * f_s_HI * np.exp(-r_distance ** 2 / (2 * self.p['sigma_arms'] ** 2)))
        n_s_CO = np.sum(
            self.p['n8_s_CO'] * f_d_CO * f_s_CO * np.exp(-r_distance ** 2 / (2 * self.p['sigma_arms'] ** 2)))

        # --- Total density ---
        self.n_HI = n_d_HI + n_s_HI
        self.n_H2 = (n_d_CO + n_b + n_s_CO) * self.p['X_CO'] * 3.24e-22

    def get_density(self):
        return self.n_HI * 1e6 * 1.67e-27 + self.n_H2 * 1e6 * 3.34 * 1e-27

    def get_element_abundance(self):
        pass

    def __str__(self):
        return f'Interstellar Gas'

    # @staticmethod
    # @jit(nopython=True, fastmath=True)  # Use numba to accelerate the function
    # def _calculate_model_static(X, Y, Z, X_CO, r_s, r_0, r_h, h_i, z_s_HI, z_s_CO, r_z0, r_z,
    #                             w_0p, w_1p, w_2p, theta_1, theta_2, n8_d_HI, n8_d_CO,
    #                             n_b, theta_b, x_0, r_b, z_b, e_i, p_i, a, r_min, theta_min,
    #                             sigma_arms, n8_s_HI, n8_s_CO):
    #     def fzero(func, x0):
    #         result = root_scalar(func, bracket=[x0 - 1, x0 + 1], method='brentq')
    #         return result.root if result.converged else -np.inf
    #
    #     theta, R = np.arctan2(Y, X), np.hypot(X, Y)
    #
    #     # --- Warped disk ---
    #     # Fit of the HI profile is passed as a function from p['f_d_HI'], so it can't be accelerated with numba
    #     # Assuming simple f_d_HI behavior here for numba compatibility; adjust as needed
    #     f_d_HI = 0.2291 * R ** 2 * np.exp(-R / 3.016) / 0.250
    #     f_d_CO = np.exp(-(R - r_s) / r_0) * (1 - np.exp(-(R / r_h) ** h_i))
    #
    #     # Warp of disk
    #     w_0 = polyval(R, w_0p)
    #     w_1 = polyval(R, w_1p)
    #     w_2 = polyval(R, w_2p)
    #     z_0 = w_0 + w_1 * np.sin(theta - theta_1) + w_2 * np.sin(2 * theta - theta_2)
    #
    #     # Vertical profile
    #     ZZ = Z - z_0
    #     z_h_HI = z_s_HI * np.exp((R - r_z0) / r_z)
    #     z_h_CO = z_s_CO * np.exp((R - r_z0) / r_z)
    #     f_s_HI = 1/np.cosh(ZZ / z_h_HI) ** 0.5
    #     f_s_CO = 1/np.cosh(ZZ / z_h_CO) ** 2.0
    #
    #     # Final disk
    #     n_d_HI = n8_d_HI * f_d_HI * f_s_HI
    #     n_d_CO = n8_d_CO * f_d_CO * f_s_CO
    #
    #     # --- Central bulge ---
    #     XX = X * np.cos(theta_b) + Y * np.sin(theta_b) + x_0
    #     YY = -X * np.sin(theta_b) + Y * np.cos(theta_b)
    #     RR = np.sqrt(XX ** 2 + (YY / 0.3) ** 2)
    #     Rr = 1 / (RR / r_b + Z / z_b)
    #     n_bulge = n_b * np.exp(-Rr ** e_i) * Rr ** p_i
    #
    #     # --- Spiral arms ---
    #     r_distance = np.zeros(4)
    #     for i in range(4):
    #         fun = lambda x, rmin, r, t, t_min, b: rmin / r * np.exp((x - t_min) / b) - np.cos(x - t) + b * np.sin(x - t)
    #         theta_guess = np.linspace(theta_min[i], a[i] * np.log(25 / r_min[i]) + theta_min[i], 100)
    #         zero_crossing = fun(theta_guess, r_min[i], R, theta, theta_min[i], a[i]) * \
    #                         np.roll(fun(theta_guess, r_min[i], R, theta, theta_min[i], a[i]), 1) <= 0
    #         zero_crossing[[0, -1]] = False
    #         theta_guess = theta_guess[zero_crossing]
    #         if theta_guess.size == 0:
    #             continue
    #         theta_candidate = np.array([fzero(lambda x: fun(x, r_min[i], R, theta, theta_min[i], a[i]), tg) for tg in
    #                                     theta_guess])
    #         r_candidate = r_min[i] * np.exp((theta_candidate - theta_min[i]) / a[i])
    #         X_candidate, Y_candidate = r_candidate * np.cos(theta_candidate), r_candidate * np.sin(theta_candidate)
    #         r_distance[i] = np.min(np.sqrt((X_candidate - X) ** 2 + (Y_candidate - Y) ** 2))
    #
    #     n_s_HI = np.sum(n8_s_HI * f_d_HI * f_s_HI * np.exp(-r_distance ** 2 / (2 * sigma_arms ** 2)))
    #     n_s_CO = np.sum(n8_s_CO * f_d_CO * f_s_CO * np.exp(-r_distance ** 2 / (2 * sigma_arms ** 2)))
    #
    #     # --- Total density ---
    #     n_HI = n_d_HI + n_s_HI  # cm**(-3)
    #     n_H2 = (n_d_CO + n_bulge + n_s_CO) * X_CO * 3.24e-22  # cm**(-3)
    #
    #     return n_HI, n_H2
