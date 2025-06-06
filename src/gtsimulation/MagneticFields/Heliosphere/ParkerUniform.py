import datetime

import numpy as np
from numba import jit, prange

from gtsimulation.Global import Units, Regions
from gtsimulation.MagneticFields.Heliosphere import Parker
from gtsimulation.MagneticFields.Heliosphere.Functions import transformations


class ParkerUniform(Parker):
    def __init__(self, x, y, z, t=None, coeff_noise=0.95, *args, **kwargs):
        super().__init__(coeff_noise=coeff_noise, *args, **kwargs)
        self.ModelName = "ParkerUniform"
        kwargs["use_noise"] = False
        self.b = Parker(*args, **kwargs)
        self.Bx, self.By, self.Bz = self.b.CalcBfield(x, y, z, t=t)
        self.x, self.y, self.z = x, y, z
        self.r, _, self.theta, self.phi = transformations.Cart2Sphere(x, y, z)
        self.wind = self.v_wind(self.theta, self.km2AU)

    def CalcBfield(self, x=None, y=None, z=None, **kwargs):
        Bx, By, Bz = 0, 0, 0
        if self.use_reg:
            Bx += self.Bx
            By += self.By
            Bz += self.Bz

        if not self.use_noise:
            return Bx, By, Bz

        r, _, theta, phi = transformations.Cart2Sphere(x, y, z)
        omega = self.omega
        rs = self.rs
        v_wind = self.wind
        a = v_wind / omega

        A_rad = self.A_rad
        alpha_rad = self.alpha_rad
        delta_rad = self.delta_rad

        A_azimuth = self.A_azimuth
        alpha_azimuth = self.alpha_azimuth
        delta_azimuth = self.delta_azimuth

        A_2D = self.A_2D
        alpha_2D = self.alpha_2D
        delta_2D = self.delta_2D

        k = self.k
        dk = self.dk

        Bx_n, By_n, Bz_n = self._calc_noise(self.r, self.theta, self.phi, a,
                                            x, y, z,
                                            A_rad, alpha_rad, delta_rad,
                                            A_azimuth, alpha_azimuth, delta_azimuth,
                                            A_2D, alpha_2D, delta_2D,
                                            rs, k, dk, self.use_slab, self.use_2d, self.coeff_2d)

        Bx += self.magnitude * self.coeff_noise * Bx_n
        By += self.magnitude * self.coeff_noise * By_n
        Bz += self.magnitude * self.coeff_noise * Bz_n

        return Bx, By, Bz

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def _calc_noise(r_param, theta_param, phi_param, a,
                    x, y, z,
                    A_rad, alpha_rad, delta_rad,
                    A_azimuth, alpha_azimuth, delta_azimuth,
                    A_2d, alpha_2d, delta_2d,
                    rs, k, dk, use_slab, use_2d, component_2d):
        """
        doi.org/10.3847/1538-4357/aca892/meta
        """
        q_slab = 5 / 3
        q_2d = 8 / 3
        p = 0
        gamma = 3

        cospsi = 1. / np.sqrt(1 + ((r_param - rs) * np.sin(theta_param) / a) ** 2)
        sinpsi = ((r_param - rs) * np.sin(theta_param) / a) / np.sqrt(1 + ((r_param - rs) * np.sin(theta_param) / a) ** 2)

        cospsi_ = 1. / np.sqrt(1 + ((r_param - rs) / a) ** 2)
        sinpsi_ = ((r_param - rs) / a) / np.sqrt(1 + ((r_param - rs) / a) ** 2)

        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        kk = np.array([0, 0, 1])

        e_r = np.sin(theta_param) * np.cos(phi_param) * i + np.sin(theta_param) * np.sin(phi_param) * j + np.cos(theta_param) * kk
        e_phi = -np.sin(phi_param) * i + np.cos(phi_param) * j
        e_theta = np.cos(theta_param) * np.cos(phi_param) * i + np.cos(theta_param) * np.sin(phi_param) * j - np.sin(theta_param) * kk

        ez = cospsi * e_r - sinpsi * e_phi
        ex = e_theta
        ey = sinpsi * e_r + cospsi * e_phi

        z_new = ez@np.array([x, y, z])
        x_new = ex@np.array([x, y, z])
        y_new = ey@np.array([x, y, z])

        lam_2d = 0.04 * (r_param / (rs / 5)) ** 0.8 * (rs / 5)
        lam_slab = 2 * lam_2d

        Bx_helio, By_helio, Bz_helio = 0., 0., 0.

        # TODO: calculation is point wise

        for mod in prange(len(k)):
            # Slab spectrum

            numer_slab = dk[mod, 0] * k[mod, 0] ** p
            B_azimuth = A_azimuth[mod, 0] * r_param ** (-gamma / 2)
            brk_azimuth = lam_slab * k[mod, 0] / r_param
            denom_azimuth = (1 + brk_azimuth ** (p + q_slab))
            spectrum_azimuth = np.sqrt(numer_slab / denom_azimuth)

            deltaB_azimuth = B_azimuth * spectrum_azimuth

            # 2d spectrum

            B_2d = A_2d[mod, 0] * r_param ** (-gamma / 2)
            brk_2d = lam_2d * k[mod, 0] / r_param
            denom_2d = (1 + brk_2d ** (p + q_2d))
            numer_2d = dk[mod, 0] * k[mod, 0] ** (p + 1)
            spectrum_2d = np.sqrt(2 * np.pi * numer_2d / denom_2d)
            deltaB_2d = B_2d * spectrum_2d

            # Slab polarization and phase
            phase_azimuth = k[mod, 0] * z_new/r_param + delta_azimuth[mod, 0]

            # 2d polarization and phase
            phase_2d = k[mod, 0]/r_param * (np.cos(alpha_2d[mod, 0]) * x_new + np.sin(alpha_2d[mod, 0]) * y_new) + \
                       delta_2d[mod, 0]

            # Slab field
            Bx_slab = deltaB_azimuth * np.cos(phase_azimuth) * np.cos(alpha_azimuth[mod, 0])
            By_slab = deltaB_azimuth * np.cos(phase_azimuth) * np.sin(alpha_azimuth[mod, 0])
            Bz_slab = 0

            # 2D field
            Bx_2d = deltaB_2d * np.cos(phase_2d) * np.cos(alpha_2d[mod, 0])
            By_2d = deltaB_2d * np.cos(phase_2d) * np.sin(alpha_2d[mod, 0])
            Bz_2d = 0


            # Total field
            coeff_slab = 0
            coeff_2d = 0
            if use_slab:
                coeff_slab = 1. #5.546/2 #1 / 2

            if use_2d:
                coeff_2d = component_2d

            Bx_helio += coeff_2d*Bx_2d + coeff_slab * Bx_slab
            By_helio += coeff_2d*By_2d + coeff_slab * By_slab
            Bz_helio += coeff_2d*Bz_2d + coeff_slab * Bz_slab

        Bx = Bx_helio * i@ex + By_helio * i@ey + Bz_helio * i@ez
        By = Bx_helio * j@ex + By_helio * j@ey + Bz_helio * j@ez
        Bz = Bx_helio * kk@ex + By_helio * kk@ey + Bz_helio * kk@ez

        return Bx * (r_param > rs), By * (r_param > rs), Bz * (r_param > rs)

    def to_string(self):
        s = super().to_string()
        s = s.replace("Parker", "ParkerUniform")
        s += f"""
        Coordinates: {self.x}, {self.y}, {self.z} AU"""
        return s
