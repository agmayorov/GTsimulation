import datetime
import numpy as np
from numba import jit, prange

from Global import Units, Regions
from MagneticFields import AbsBfield
from MagneticFields.Heliosphere.Functions import transformations

from timeit import default_timer as timer


class Parker(AbsBfield):
    ToMeters = Units.AU2m
    rs = 0.0232523
    omega = 2 * np.pi / 2160000
    years11 = 347133600
    km2AU = 1 / Units.AU2km

    def __init__(self, date: int | datetime.date = 0, magnitude=2.09, use_reg=True, use_cir=False, polarity=-1,
                 use_noise=False, noise_num=256, log_kmin=0, log_kmax=6, coeff2d=1.4, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Heliosphere
        self.ModelName = "Parker"
        self.Units = "AU"
        self.magnitude = magnitude
        self.use_cir = use_cir
        self.polarity = polarity
        self.use_reg = use_reg
        self.coeff2d = coeff2d
        self.__set_time(date)
        self.__set_noise(use_noise, noise_num, log_kmin, log_kmax)

    def __set_time(self, date: int | datetime.datetime):
        self.Date = date
        self.t = 2488320  # To have a correct phase with ace data
        if isinstance(date, int):
            self.t += date
            return
        year = date.year
        doy = date.timetuple().tm_yday
        hour = date.hour
        minute = date.minute
        second = date.second
        self.t += (((year * 365.25 + doy) * 24 + hour) * 60 + minute) * 60 + second

    def CalcBfield(self, x, y, z, **kwargs):
        if kwargs.get("t") is not None:
            self.t = kwargs.get("t")

        A0 = self.magnitude * self.polarity
        t = self.t
        r, R, theta, phi = transformations.Cart2Sphere(x, y, z)
        v_wind = self.v_wind(theta, self.km2AU)
        omega = self.omega
        rs = self.rs

        Bx = np.zeros_like(r)
        By = np.zeros_like(r)
        Bz = np.zeros_like(r)

        if self.use_reg:
            years11 = self.years11

            alpha = self.CalcTiltAngle(t)
            dalpha = np.sign(self.CalcTiltAngle(t + 1) - self.CalcTiltAngle(t - 1))

            Br, Bphi = self._calc_regular(A0, t, r, theta, phi, v_wind, omega, rs, years11, alpha, dalpha)

            # alpha -= np.pi / years11 * (r - rs) / v_wind * dalpha
            #
            # theta0 = np.pi / 2 - np.arctan(-np.tan(alpha) * np.sin(phi + omega * (r - rs) / v_wind -
            #                                                        omega * t))
            #
            # HCS = self.HCS(theta, theta0, r) * (r >= rs)
            # Br = A0 / r ** 2 * HCS
            # Bphi = -A0 / r ** 2 * (((r - rs) * omega) / v_wind) * np.sin(theta) * HCS

            Bx, By, Bz = transformations.Sphere2Cart(Br, 0, Bphi, theta, phi)

        if self.use_cir:
            # TODO: add CIR
            pass

        if not self.use_noise:
            return Bx, By, Bz

        # coeff2d = 1.4
        # coeffslab = coeff2d / 2

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

        Bx_n, By_n, Bz_n = self._calc_noise(r, theta, phi, a,
                                            A_rad, alpha_rad, delta_rad,
                                            A_azimuth, alpha_azimuth, delta_azimuth,
                                            A_2D, alpha_2D, delta_2D,
                                            rs, k, dk)

        Bx += self.magnitude * self.coeff2d * Bx_n
        By += self.magnitude * self.coeff2d * By_n
        Bz += self.magnitude * self.coeff2d * Bz_n

        return Bx, By, Bz

    def UpdateState(self, new_date):
        self.__set_time(new_date)

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def _calc_regular(A0, t, r, theta, phi, v_wind, omega, rs, years11, alpha, dalpha):
        alpha_n = alpha - np.pi / years11 * (r - rs) / v_wind * dalpha

        theta0 = np.pi / 2 - np.arctan(-np.tan(alpha_n) * np.sin(phi + omega * (r - rs) / v_wind -
                                                                 omega * t))
        L = 0.0002
        dt = r * (theta - theta0) / L
        HCS = -np.tanh(dt)

        Br = A0 / r ** 2 * HCS
        Bphi = -A0 / r ** 2 * (((r - rs) * omega) / v_wind) * np.sin(theta) * HCS

        return Br, Bphi

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def CalcTiltAngle(t):
        a0 = 0.7502
        a1 = 0.02332
        b1 = -0.01626
        a2 = -0.3268
        b2 = 0.2016
        a3 = -0.02814
        b3 = 0.0005215
        a4 = -0.08341
        b4 = -0.04852
        w = 9.318e-09

        alpha = (a0 +
                 a1 * np.cos(t * w) + b1 * np.sin(t * w) +
                 a2 * np.cos(2 * t * w) + b2 * np.sin(2 * t * w) +
                 a3 * np.cos(3 * t * w) + b3 * np.sin(3 * t * w) +
                 a4 * np.cos(4 * t * w) + b4 * np.sin(4 * t * w))

        return alpha

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def v_wind(theta, km2AU):
        return (300 + 475 * (1 - np.sin(theta) ** 8)) * km2AU

    @classmethod
    def a(cls, theta):
        return cls.v_wind(theta) / cls.omega

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def HCS(theta, theta0, r):
        L = 0.0002
        dt = r * (theta - theta0) / L
        return -np.tanh(dt)

    def __set_noise(self, use_noise, noise_num, log_kmin, log_kmax):
        self.use_noise = use_noise
        if self.use_noise is False:
            return

        self.noise_num = noise_num
        self.log_kmin = log_kmin
        self.log_kmax = log_kmax

        self.k = np.logspace(log_kmin, log_kmax, self.noise_num)[:, np.newaxis]
        self.dk = self.k * (10 ** ((log_kmax - log_kmin) / (self.noise_num - 1)) - 1)

        self.A_2D = np.random.randn(self.noise_num, 1) / 130
        self.alpha_2D = np.random.rand(self.noise_num, 1) * 2 * np.pi
        n = np.trunc(np.sin(self.alpha_2D) * self.k)
        self.alpha_2D = np.real(np.arcsin(n / self.k) * (np.cos(self.alpha_2D) > 0) +
                                (np.pi - np.arcsin(n / self.k)) * (np.cos(self.alpha_2D) < 0))
        self.delta_2D = np.random.rand(self.noise_num, 1) * 2 * np.pi

        self.A_rad = np.random.randn(self.noise_num, 1) / 1.5
        self.alpha_rad = np.random.rand(self.noise_num, 1) * 2 * np.pi
        n = np.trunc(np.sin(self.alpha_rad) * self.k)
        self.alpha_rad = np.real(np.arcsin(n / self.k) * (np.cos(self.alpha_rad) > 0) +
                                 (np.pi - np.arcsin(n / self.k)) * (np.cos(self.alpha_rad) < 0))
        self.delta_rad = np.random.rand(self.noise_num, 1) * 2 * np.pi

        self.A_azimuth = np.random.randn(self.noise_num, 1) / 4.5
        self.alpha_azimuth = np.random.rand(self.noise_num, 1) * 2 * np.pi

        n = np.trunc(np.sin(self.alpha_azimuth) * self.k)
        self.alpha_azimuth = np.real(np.arcsin(n / self.k) * (np.cos(self.alpha_azimuth) > 0) +
                                     (np.pi - np.arcsin(n / self.k)) * (np.cos(self.alpha_azimuth) < 0))
        self.delta_azimuth = np.random.rand(self.noise_num, 1) * 2 * np.pi

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def _calc_noise(r, theta, phi, a,
                    A_rad, alpha_rad, delta_rad,
                    A_azimuth, alpha_azimuth, delta_azimuth,
                    A_2d, alpha_2d, delta_2d,
                    rs, k, dk):
        q_slab = 5 / 3
        q_2d = 8 / 3
        p = 0
        gamma = 3

        cospsi = 1. / np.sqrt(1 + (r * np.sin(theta) / a) ** 2)
        sinpsi = (r * np.sin(theta) / a) / np.sqrt(1 + (r * np.sin(theta) / a) ** 2)

        lam_2d = 0.04 * (r / rs) ** 0.8 * rs
        lam_slab = 2 * lam_2d

        Br, Btheta, Bphi = 0., 0., 0.

        # TODO: calculation is point wise

        for mod in prange(len(k)):
            numer_slab = dk[mod, 0] * k[mod, 0] ** p

            # Radial spectrum

            B_rad = A_rad[mod, 0] * r ** (-gamma / 2)
            brk_rad = lam_slab * k[mod, 0] / np.sqrt(a * r)
            denom_rad = (1 + brk_rad ** (p + q_slab))

            spectrum_rad = np.sqrt(numer_slab / denom_rad)
            deltaB_rad = 2 * B_rad * spectrum_rad * cospsi * r * np.sqrt(r * a)

            # Azimuthal spectrum

            B_azimuth = A_azimuth[mod, 0] * r ** (-gamma / 2)
            brk_azimuth = lam_slab * k[mod, 0] / r
            denom_azimuth = (1 + brk_azimuth ** (p + q_slab))
            spectrum_azimuth = np.sqrt(numer_slab / denom_azimuth)

            deltaB_azimuth = B_azimuth * spectrum_azimuth
            dpsectrum_azimuth = (0.08 * (p + q_slab) * numer_slab * brk_azimuth ** (p + q_slab - 1) /
                                 (2 * spectrum_azimuth * (denom_azimuth ** 2)) * (
                                         0.8 * k[mod, 0] / (r * (r / rs) ** 0.2)
                                         - brk_azimuth / (0.08 * r)))

            # 2d spectrum

            lam_2d = 0.04 * (r / rs) ** 0.8 * rs
            B_2d = A_2d[mod, 0] * r ** (-gamma / 2)
            brk_2d = lam_2d * k[mod, 0] / r
            denom_2d = (1 + brk_2d ** (p + q_2d))
            numer_2d = dk[mod, 0] * k[mod, 0] ** (p + 1)
            spectrum_2d = np.sqrt(2 * np.pi * numer_2d) / np.sqrt(denom_2d)
            deltaB_2d = B_2d * spectrum_2d

            dspectrum_2d = (-(0.04 * np.sqrt(np.pi / 2) * (p + q_2d) * numer_2d * brk_2d ** (p + q_2d - 1)) /
                            (spectrum_2d / np.sqrt(2 * np.pi) * denom_2d ** 2) *
                            (0.8 * k[mod, 0] / (r * (r / rs) ** 0.2) - brk_2d / (0.04 * r)))

            # Radial polarization and phase
            phase_rad = k[mod, 0] * np.sqrt(r / a) + delta_rad[mod, 0]

            # Azimuthal polarization and phase
            phase_azimuth = k[mod, 0] * phi + delta_azimuth[mod, 0]

            # 2d polarization and phase
            phase_2d = k[mod, 0] * (r / a + phi + theta * np.cos(alpha_2d[mod, 0])) + delta_2d[mod, 0]

            # Radial field
            Br_rad = 0

            Btheta_rad = (-deltaB_rad * np.cos(phase_rad) * 1 / np.sin(theta) * np.sin(alpha_rad[mod, 0]) /
                          (2 * r * np.sqrt(r / a)))

            Bphi_rad = (deltaB_rad / r * np.cos(alpha_rad[mod, 0]) * cospsi * np.cos(phase_rad) /
                        (2 * r * np.sqrt(r / a)))

            # Azimuthal field

            Br_az = -deltaB_azimuth * sinpsi * np.cos(alpha_azimuth[mod, 0]) * np.cos(phase_azimuth)
            Btheta_az = deltaB_azimuth * sinpsi * np.sin(alpha_azimuth[mod, 0]) * np.cos(phase_azimuth)
            Bphi_az = (deltaB_azimuth * (1 - gamma / 2) * np.cos(alpha_azimuth[mod, 0]) * np.sin(theta) *
                       sinpsi * np.sin(phase_azimuth) -
                       deltaB_azimuth * sinpsi * np.sin(phase_azimuth) *
                       (np.cos(theta) * np.sin(alpha_azimuth[mod, 0]) -
                        np.sin(theta) * np.cos(alpha_azimuth[mod, 0])) +
                       B_azimuth / r * dpsectrum_azimuth * np.cos(alpha_azimuth[mod, 0]) *
                       np.sin(theta) * np.sin(phase_azimuth) * sinpsi) / k[mod, 0]

            # 2d field
            Br_2d = (-deltaB_2d / r * sinpsi * (np.cos(alpha_2d[mod, 0]) * np.cos(phase_2d) +
                                                (1 / np.tan(theta) * np.sin(phase_2d)) / k[mod, 0]))

            Btheta_2d = (deltaB_2d * (sinpsi * np.cos(phase_2d) / a -
                                      gamma * sinpsi * np.sin(phase_2d) / (2 * r * k[mod, 0]) +
                                      1 / np.sin(theta) * (cospsi * np.cos(phase_2d) +
                                                           np.sin(theta) * sinpsi * np.sin(phase_2d) / k[mod, 0])) +
                         B_2d * sinpsi * np.sin(phase_2d) * dspectrum_2d / k[mod, 0])

            Bphi_2d = -deltaB_2d / r * np.cos(alpha_2d[mod, 0]) * cospsi * np.cos(phase_2d)

            # Total field
            coeff_slab = 1 / 2

            Br += Br_2d + coeff_slab * (Br_az + Br_rad)
            Btheta += Btheta_2d + coeff_slab * (Btheta_az + Btheta_rad)
            Bphi += Bphi_2d + coeff_slab * (Bphi_az + Bphi_rad)

        # B = np.zeros((3, *Br_2d.shape))
        # B[0] = Br_2d + coeff_slab * (Br_az + Br_rad)
        # B[1] = Btheta_2d + coeff_slab * (Btheta_az + Btheta_rad)
        # B[2] = Bphi_2d + coeff_slab * (Bphi_az + Bphi_rad)
        # B_s = np.sum(B, axis=1)
        # Br, Btheta, Bphi = B_s[0], B_s[1], B_s[2]

        # Br = np.sum(Br_2d + coeff_slab * (Br_az + Br_rad), axis=0)
        # Btheta = np.sum(Btheta_2d + coeff_slab * (Btheta_az + Btheta_rad), axis=0)
        # Bphi = np.sum(Bphi_2d + coeff_slab * (Bphi_az + Bphi_rad), axis=0)

        Bx, By, Bz = transformations.Sphere2Cart(Br, Btheta, Bphi, theta, phi)
        return Bx * (r > rs), By * (r > rs), Bz * (r > rs)

    def __str__(self):
        s = f"""Parker
        Regular: {self.use_reg}
        Magnitude: {self.magnitude}
        CIR: {self.use_cir}
        Polarity: {self.polarity}
        Noise: {self.use_noise}"""

        if self.use_noise:
            s += f"""
            Min wave length: {self.log_kmin}
            Max wave length: {self.log_kmax}
            Number of waves: {self.noise_num}
            Coeff_2d: {self.coeff2d}"""

        return s


if __name__ == "__main__":
    from datetime import datetime
    import matplotlib.pyplot as plt


    b = Parker(date=datetime(2008, 1, 1), use_noise=False)
    b0 = Parker(date=datetime(2008, 1, 1), use_noise=True, log_kmin=0, log_kmax=6, num=256)
    b1 = Parker(date=datetime(2008, 1, 1), use_noise=True, log_kmin=1, log_kmax=6, num=256)
    z=0.
    xx = np.cos(np.arange(0, 6*np.pi, 0.1))
    yy = np.sin(np.arange(0, 6*np.pi, 0.1))
    B0 = []
    B1 = []
    B = []
    for x, y in zip(xx, yy):
        B.append(b.CalcBfield(x, y, z))
        B0.append(b0.CalcBfield(x, y, z))
        B1.append(b1.CalcBfield(x, y, z))
    B = np.array(B)
    B0 = np.array(B0)
    B1 = np.array(B1)

    # plt.plot(np.arange(0, 6*np.pi, 0.1), B[:,0]*xx + B[:, 1]*yy, label='reg')
    # plt.plot(np.arange(0, 6*np.pi, 0.1), B0[:,0]*xx + B0[:, 1]*yy, label=0)
    plt.plot(np.arange(0, 6*np.pi, 0.1), B0[:,2], label='z0')
    plt.plot(np.arange(0, 6*np.pi, 0.1), B1[:,2], label='z1')
    # plt.plot(np.arange(0, 6*np.pi, 0.1), B1[:,0]*xx + B1[:, 1]*yy, label=1)
    plt.legend()

    plt.show()

    # st = timer()
    # b.CalcBfield(40, 50, 60)
    # print(timer() - st)
