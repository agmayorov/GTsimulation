import numpy as np
from numpy.polynomial.polynomial import polyval
from numba import jit

from Global import Regions, Units
from Medium import GTGeneralMedium


class InterstellarGas(GTGeneralMedium):
    def __init__(self):
        super().__init__()
        self.region = Regions.Galaxy
        self.model = "Interstellar Gas"
        self.n_HI = 0
        self.n_H2 = 0

    def calculate_model(self, x, y, z, **kwargs):
        self.n_HI, self.n_H2 = self.__calculate_b_field(x / Units.kpc, y / Units.kpc, z / Units.kpc)

    def get_density(self):
        return self.n_HI * 1e6 * 1.67e-27 + self.n_H2 * 1e6 * 2 * 1.67e-27 # kg/m3

    def get_element_list(self):
        return ['H']

    def get_element_abundance(self):
        return np.array([1])

    def to_string(self):
        return self.model

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __calculate_b_field(x, y, z):
        X_CO = 2e20

        # Radial profile for disk and arms
        r_s = 8.0
        r_0 = 1.27
        r_h = 6.34
        h_i = 6.38
        n8_d_HI = 0.160
        n8_d_CO = 0.894

        # Warped disk
        z_s_HI = 0.0942
        z_s_CO = 0.103
        r_z0 = 8.5
        r_z = 6.94
        w_0p = np.array([-4.759003e-2, 3.499272e-3, -2.426709e-3, 2.075231e-4])
        w_1p = np.array([1.412557e-1, -7.066979e-2, 6.260557e-3, 6.192461e-5])
        w_2p = np.array([2.830553e-1, -6.618279e-2, 1.885424e-3, 1.057025e-4])
        theta_1 = 4.61
        theta_2 = 2.73

        # Central bulge
        n_b = 47.8
        theta_b = 5.67
        x_0 = 0.751
        r_b = 0.514
        z_b = 6.43e-3 # kpc
        e_i = 0.647
        p_i = 1.18

        # Spiral arms
        a = np.array([3.30, 4.35, 5.32, 4.75])
        r_min = np.array([2.00, 3.31, 3.89, 3.19])
        theta_min = -np.pi / 6 + np.array([1, 2, 3, 4]) * np.pi / 2
        sigma_arms = 0.6
        n8_s_HI = np.array([0.184, 0.193, 0.332, 0.521])
        n8_s_CO = np.array([0.642, 0, 3.37, 7.53])

        r_max = 25
        theta_max = a * np.log(r_max / r_min) + theta_min

        def spiral(phi, r_min, phi_min, alpha):
            return r_min * np.exp((phi - phi_min) / alpha)

        def get_distance(phi, r, r_min, r_max, phi_min, phi_max, alpha):
            dist = r_max
            while phi < phi_max:
                if phi >= phi_min:
                    dist_new = np.abs(r - spiral(phi, r_min, phi_min, alpha))
                    if dist_new < dist:
                        dist = dist_new
                phi += 2 * np.pi
            return dist * np.cos(np.arctan(1 / alpha))

        # Convert cartesian to polar coordinates
        theta, r = np.arctan2(y, x), np.sqrt(x ** 2 + y ** 2)
        theta += np.pi # rotate the model so that the sun is at the point [-8.5, 0, 0] kpc

        # --- Warped disk ---
        # Radial profile
        f_d_HI = 0.075 * r ** 3 * np.exp(-r / 2.235) # normalized on n8
        f_d_CO = np.exp(-(r - r_s) / r_0) * (1 - np.exp(-(r / r_h) ** h_i))

        # Warp of disk
        w_0 = polyval(r, w_0p)
        w_1 = polyval(r, w_1p)
        w_2 = polyval(r, w_2p)
        z_0 = w_0 + w_1 * np.sin(theta - theta_1) + w_2 * np.sin(2 * theta - theta_2)

        # Vertical profile
        zz = z - z_0
        z_h_HI = z_s_HI * np.exp((r - r_z0) / r_z)
        z_h_CO = z_s_CO * np.exp((r - r_z0) / r_z)
        f_s_HI = 1 / np.cosh(zz / z_h_HI) ** 0.5
        f_s_CO = 1 / np.cosh(zz / z_h_CO) ** 2.0

        # Final disk
        n_d_HI = n8_d_HI * f_d_HI * f_s_HI
        n_d_CO = n8_d_CO * f_d_CO * f_s_CO

        # --- Central bulge ---
        xx = x * np.cos(theta_b) + y * np.sin(theta_b) + x_0
        yy = -x * np.sin(theta_b) + y * np.cos(theta_b)
        rr = np.sqrt(xx ** 2 + (yy / 0.3) ** 2)
        Rr = 1 / (rr / r_b + np.abs(z) / z_b)
        n_bulge = n_b * np.exp(-Rr ** e_i) * Rr ** p_i

        # --- Spiral arms ---
        r_distance = np.zeros(4)
        for i in range(4):
            r_distance[i] = get_distance(theta, r, r_min[i], r_max, theta_min[i], theta_max[i], a[i])
        n_s_HI = np.nansum(n8_s_HI * f_d_HI * f_s_HI * np.exp(-r_distance ** 2 / (2 * sigma_arms ** 2)))
        n_s_CO = np.nansum(n8_s_CO * f_d_CO * f_s_CO * np.exp(-r_distance ** 2 / (2 * sigma_arms ** 2)))

        # --- Total density ---
        n_HI = n_d_HI + n_s_HI  # cm**(-3)
        n_H2 = (n_d_CO + n_bulge + n_s_CO) * X_CO * 3.24e-22  # cm**(-3)

        return n_HI, n_H2
