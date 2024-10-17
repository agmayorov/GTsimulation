import datetime

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange

from Global import Units, Regions
from MagneticFields.Heliosphere import Parker
from MagneticFields.Heliosphere.Functions import transformations


class ParkerUniform(Parker):
    def __init__(self, x, y, z, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
                                            r, theta, phi,
                                            A_rad, alpha_rad, delta_rad,
                                            A_azimuth, alpha_azimuth, delta_azimuth,
                                            A_2D, alpha_2D, delta_2D,
                                            rs, k, dk, self.use_slab, self.use_2d)

        Bx += self.magnitude * self.coeff2d * Bx_n
        By += self.magnitude * self.coeff2d * By_n
        Bz += self.magnitude * self.coeff2d * Bz_n

        return Bx, By, Bz

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def _calc_noise(r_param, theta_param, phi_param, a,
                    r, theta, phi,
                    A_rad, alpha_rad, delta_rad,
                    A_azimuth, alpha_azimuth, delta_azimuth,
                    A_2d, alpha_2d, delta_2d,
                    rs, k, dk, use_slab, use_2d):
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

        lam_2d = 0.04 * (r_param / (rs / 5)) ** 0.8 * (rs / 5)
        dlamd_2d = 0.032 * (rs / (5 * r_param)) ** 0.2
        lam_slab = 2 * lam_2d
        dlam_slab = 2 * dlamd_2d

        Br, Btheta, Bphi = 0., 0., 0.

        # TODO: calculation is point wise

        for mod in prange(len(k)):
            numer_slab = dk[mod, 0] * k[mod, 0] ** p

            # Radial spectrum

            B_rad = A_rad[mod, 0] * r_param ** (-gamma / 2)
            brk_rad = lam_slab * k[mod, 0] / np.sqrt(a * r_param)
            denom_rad = (1 + brk_rad ** (p + q_slab))

            spectrum_rad = np.sqrt(numer_slab / denom_rad)
            deltaB_rad = 2 * B_rad * spectrum_rad * cospsi_ * r_param * np.sqrt(r_param * a)

            # Azimuthal spectrum

            B_azimuth = A_azimuth[mod, 0] * r_param ** (-gamma / 2)
            brk_azimuth = lam_slab * k[mod, 0] / r_param
            denom_azimuth = (1 + brk_azimuth ** (p + q_slab))
            spectrum_azimuth = np.sqrt(numer_slab / denom_azimuth)

            deltaB_azimuth = B_azimuth * spectrum_azimuth
            dspectrum_azimuth = -spectrum_azimuth * (p + q_2d) * (denom_azimuth - 1) * (r_param * dlam_slab - lam_slab) / (
                    denom_azimuth * 2 * r_param * lam_slab)
            ddeltaB_azimtuth = B_azimuth * dspectrum_azimuth + spectrum_azimuth * B_azimuth * (-gamma / (2 * r_param))

            # 2d spectrum

            B_2d = A_2d[mod, 0] * r_param ** (-gamma / 2)
            brk_2d = lam_2d * k[mod, 0] / r_param
            denom_2d = (1 + brk_2d ** (p + q_2d))
            numer_2d = dk[mod, 0] * k[mod, 0] ** (p + 1)
            spectrum_2d = np.sqrt(2 * np.pi * numer_2d / denom_2d)
            deltaB_2d = B_2d * spectrum_2d

            dspectrum_2d = -spectrum_2d * (p + q_2d) * (denom_2d - 1) * (r_param * dlamd_2d - lam_2d) / (
                    denom_2d * 2 * r_param * lam_2d)
            ddeltaB_2d = B_2d * dspectrum_2d + spectrum_2d * B_2d * (-gamma / (2 * r_param))

            # Radial polarization and phase
            phase_rad = k[mod, 0] * np.sqrt(r / a) + delta_rad[mod, 0]

            # Azimuthal polarization and phase
            phase_azimuth = k[mod, 0] * phi + delta_azimuth[mod, 0]

            # 2d polarization and phase
            phase_2d = k[mod, 0] * ((r / a + phi) * np.sin(alpha_2d[mod, 0]) + theta * np.cos(alpha_2d[mod, 0])) + \
                       delta_2d[mod, 0]

            # Radial field
            Br_rad = 0
            Btheta_rad = -deltaB_rad * a * np.sin(alpha_rad[mod, 0]) * np.cos(phase_rad) / (
                    2 * r_param * np.sin(theta_param) * np.sqrt(a * r_param))
            Bphi_rad = deltaB_rad * a * np.cos(alpha_rad[mod, 0]) * np.cos(phase_rad) / (
                    2 * r_param * np.sin(theta_param) * np.sqrt(a * r_param))

            # Azimuthal field

            Br_az = -deltaB_azimuth * sinpsi_ * np.cos(alpha_azimuth[mod, 0]) * np.cos(phase_azimuth)
            Btheta_az = deltaB_azimuth * sinpsi_ * np.sin(alpha_azimuth[mod, 0]) * np.cos(phase_azimuth)
            Bphi_az = 1/k[mod, 0] * (np.sin(theta_param) * np.sin(phase_azimuth) * np.cos(alpha_azimuth[mod, 0]) *
                                     (2*deltaB_azimuth*sinpsi_ + r_param/a * deltaB_azimuth * cospsi_ + r_param * sinpsi_ * ddeltaB_azimtuth) -
                                     np.cos(theta_param)*deltaB_azimuth*np.sin(phase_azimuth)*sinpsi_*np.sin(alpha_azimuth[mod, 0]))

            # 2d field
            Br_2d = -deltaB_2d / (r_param * k[mod, 0]) * (np.sin(phase_2d)*sinpsi*np.tan(theta_param)**(-1) +
                                                     k[mod, 0]*np.cos(alpha_2d[mod, 0])*np.cos(phase_2d)*sinpsi +
                                                     np.sin(phase_2d)*sinpsi*cospsi**2*np.tan(theta_param)**(-1))
            Btheta_2d = deltaB_2d / (r_param * np.sin(theta_param)) * cospsi * np.sin(alpha_2d[mod, 0] * np.cos(phase_2d)) \
                        - np.sin(theta_param) * cospsi / (a * r_param * k[mod, 0]) * (ddeltaB_2d * r_param * (r_param - rs) * np.sin(phase_2d) +
                                                                          deltaB_2d * np.sin(phase_2d) * (
                                                                                  2 * r_param - rs - r_param * sinpsi ** 2) +
                                                                          k[mod, 0] * r_param * (r_param - rs) / a * np.sin(
                        alpha_2d[mod, 0] * np.cos(phase_2d) * deltaB_2d))

            Bphi_2d = -deltaB_2d / (r_param * k[mod, 0]) * (cospsi * k[mod, 0] * np.cos(alpha_2d[mod, 0]) * np.cos(phase_2d) -
                                                      (np.tan(theta_param))**(-1) * np.sin(phase_2d) * cospsi * sinpsi**2)

            # Total field
            coeff_slab = 0
            coeff_2d = 0
            if use_slab:
                coeff_slab = 1 / 2

            if use_2d:
                coeff_2d = 1

            Br += coeff_2d*Br_2d + coeff_slab * (Br_az + Br_rad)
            Btheta += coeff_2d*Btheta_2d + coeff_slab * (Btheta_az + Btheta_rad)
            Bphi += coeff_2d*Bphi_2d + coeff_slab * (Bphi_az + Bphi_rad)

        Bx, By, Bz = transformations.Sphere2Cart(Br, Btheta, Bphi, theta, phi)
        return Bx * (r > rs), By * (r > rs), Bz * (r > rs)

    def __str__(self):
        s = super().__str__()
        s = s.replace("Parker", "ParkerUniform")
        return s


if __name__ == "__main__":
    b = ParkerUniform(1/np.sqrt(2), 1/np.sqrt(2), 0, use_noise=True)
    b_reg = ParkerUniform(1/np.sqrt(2), 1/np.sqrt(2), 0, use_noise=False)
    y = 1/np.sqrt(2)
    z = 0
    x = 1/np.sqrt(2) + np.linspace(-.5, 0.5, 500)
    Br = []
    Br_reg = []
    for xx in x:
        Bx, By, Bz = b.CalcBfield(xx, y, z)
        r = np.sqrt(xx**2 + y**2 + z**2)
        Br.append(Bx)
        Bx, By, Bz = b_reg.CalcBfield(xx, y, z)
        Br_reg.append(Bx)

    plt.plot(x, Br)
    plt.plot(x, Br_reg)
    plt.xlabel("x, au")
    plt.ylabel("Bx, nT")
    plt.show()