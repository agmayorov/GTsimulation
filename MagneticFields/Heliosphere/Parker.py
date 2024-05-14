import datetime
import numpy as np
from numba import jit

from MagneticFields import AbsBfield, Units, Regions
from MagneticFields.Heliosphere.Functions import transformations


class Parker(AbsBfield):
    ToMeters = Units.AU2m
    rs = 0.0232523
    omega = 2 * np.pi / 2160000
    years11 = 347133600
    km2AU = 1 / Units.AU2km

    def __init__(self, date: int | datetime.date = 0, magnitude=2.09, use_cir=False, polarization=-1, use_noise=False,
                 noise_num=1024, log_kmin=0, log_kmax=7, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Heliosphere
        self.ModelName = "Parker"
        self.Units = "AU"
        self.magnitude = magnitude
        self.use_cir = use_cir
        self.polarization = polarization
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

        A0 = self.magnitude * self.polarization
        t = self.t
        r, R, theta, phi = transformations.Cart2Sphere(x, y, z)
        v_wind = self.v_wind(theta, self.km2AU)
        omega = self.omega
        rs = self.rs
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

        if self.use_cir:
            # TODO: add CIR
            pass

        Bx, By, Bz = transformations.Sphere2Cart(Br, 0, Bphi, theta, phi)

        if not self.use_noise:
            return Bx, By, Bz

        coeff2d = 1.4
        coeffslab = coeff2d / 2

        a = v_wind / omega
        num = self.noise_num
        log_kmin = self.log_kmin
        log_kmax = self.log_kmax

        A_rad = self.A_rad
        alpha_rad = self.alpha_rad
        delta_rad = self.delta_rad

        A_azimuth = self.A_azimuth
        alpha_azimuth = self.alpha_azimuth
        delta_azimuth = self.delta_azimuth

        A_2D = self.A_2D
        alpha_2D = self.alpha_2D
        delta_2D = self.delta_2D

        Bx_slab, By_slab, Bz_slab = self.CalcSlab(r, theta, phi, a, num, log_kmin, log_kmax, A_rad, alpha_rad,
                                                  delta_rad,
                                                  A_azimuth, alpha_azimuth, delta_azimuth, rs)
        Bx_2d, By_2d, Bz_2d = self.Calc2D(r, theta, phi, a, num, log_kmin, log_kmax, A_2D, alpha_2D, delta_2D, rs)

        Bx += self.magnitude * coeff2d * Bx_2d + self.magnitude * coeffslab * Bx_slab
        By += self.magnitude * coeff2d * By_2d + self.magnitude * coeffslab * By_slab
        Bz += self.magnitude * coeff2d * Bz_2d + self.magnitude * coeffslab * Bz_slab

        return Bx, By, Bz

    def UpdateState(self, new_date):
        self.__set_time(new_date)

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def _calc_regular(A0, t, r, theta, phi, v_wind, omega, rs, years11, alpha, dalpha):
        alpha -= np.pi / years11 * (r - rs) / v_wind * dalpha

        theta0 = np.pi / 2 - np.arctan(-np.tan(alpha) * np.sin(phi + omega * (r - rs) / v_wind -
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

        self.A_2D = np.random.randn(self.noise_num, 1) / 130
        self.alpha_2D = np.random.rand(self.noise_num, 1) * 2 * np.pi
        self.delta_2D = np.random.rand(self.noise_num, 1) * 2 * np.pi

        self.A_rad = np.random.randn(self.noise_num, 1) / 1.5
        self.alpha_rad = np.random.rand(self.noise_num, 1) * 2 * np.pi
        self.delta_rad = np.random.rand(self.noise_num, 1) * 2 * np.pi

        self.A_azimuth = np.random.randn(self.noise_num, 1) / 4.5
        self.alpha_azimuth = np.random.rand(self.noise_num, 1) * 2 * np.pi
        self.delta_azimuth = np.random.rand(self.noise_num, 1) * 2 * np.pi

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def CalcSlab(r, theta, phi, a, num, log_kmin, log_kmax, A_rad, alpha_rad, delta_rad,
                 A_azimuth, alpha_azimuth, delta_azimuth, rs):
        q = 5 / 3
        p = 0
        gamma = 3

        cospsi = 1. / np.sqrt(1 + (r * np.sin(theta) / a) ** 2)
        sinpsi = (r * np.sin(theta) / a) / np.sqrt(1 + (r * np.sin(theta) / a) ** 2)

        k = np.logspace(log_kmin, log_kmax, num)[:, np.newaxis]
        dk = k * (10 ** ((log_kmax - log_kmin) / (num - 1)) - 1)

        lam = 0.08 * (r / rs) ** 0.8 * rs
        numer = dk * k ** p

        # Radial spectrum

        B_rad = A_rad * r ** (-gamma / 2)
        brk_rad = lam * k / np.sqrt(a * r)
        denom_rad = (1 + brk_rad ** (p + q))

        spectrum_rad = np.sqrt(numer / denom_rad)
        deltaB_rad = 2 * B_rad * spectrum_rad * cospsi * r * np.sqrt(r * a)

        # Azimuthal spectrum

        B_azimuth = A_azimuth * r ** (-gamma / 2)
        brk_azimuth = lam * k / r
        denom_azimuth = (1 + brk_azimuth ** (p + q))
        spectrum_azimuth = np.sqrt(numer / denom_azimuth)

        deltaB_azimuth = B_azimuth * spectrum_azimuth
        dpsectrum_azimuth = (0.08 * (p + q) * numer * brk_azimuth ** (p + q - 1) /
                             (2 * spectrum_azimuth * (denom_azimuth ** 2)) * (0.8 * k / (r * (r / rs) ** 0.2)
                                                                              - brk_azimuth / (0.08 * r)))

        # Radial polarization and phase

        n = np.trunc(np.sin(alpha_rad) * k)
        alpha_rad = np.real(np.arcsin(n / k) * (np.cos(alpha_rad) > 0) + (np.pi -
                                                                          np.arcsin(n / k)) * (np.cos(alpha_rad) < 0))

        phase_rad = k * np.sqrt(r / a) + delta_rad

        # Azimuthal polarization and phase

        n = np.trunc(np.sin(alpha_azimuth) * k)
        alpha_azimuth = np.real(np.arcsin(n / k) * (np.cos(alpha_azimuth) > 0) +
                                (np.pi - np.arcsin(n / k)) * (np.cos(alpha_azimuth) < 0))

        phase_azimuth = k * phi + delta_azimuth

        # Radial field
        Br_rad = 0

        Btheta_rad = (-deltaB_rad * np.cos(phase_rad) * 1 / np.sin(theta) * np.sin(alpha_rad) /
                      (2 * r * np.sqrt(r / a)))

        Bphi_rad = (deltaB_rad / r * np.cos(alpha_rad) * cospsi * np.cos(phase_rad) /
                    (2 * r * np.sqrt(r / a)))

        # Azimuthal field

        Br_az = -deltaB_azimuth * sinpsi * np.cos(alpha_azimuth) * np.cos(phase_azimuth)
        Btheta_az = deltaB_azimuth * sinpsi * np.sin(alpha_azimuth) * np.cos(phase_azimuth)
        Bphi_az = (deltaB_azimuth * (1 - gamma / 2) * np.cos(alpha_azimuth) * np.sin(theta) *
                   sinpsi * np.sin(phase_azimuth) -
                   deltaB_azimuth * sinpsi * np.sin(phase_azimuth) *
                   (np.cos(theta) * np.sin(alpha_azimuth) -
                    np.sin(theta) * np.cos(alpha_azimuth)) +
                   B_azimuth / r * dpsectrum_azimuth * np.cos(alpha_azimuth) *
                   np.sin(theta) * np.sin(phase_azimuth) * sinpsi) / k

        # Total field
        Br = np.sum(Br_az + Br_rad, axis=0)
        Btheta = np.sum(Btheta_az + Btheta_rad, axis=0)
        Bphi = np.sum(Bphi_az + Bphi_rad, axis=0)

        Bx, By, Bz = transformations.Sphere2Cart(Br, Btheta, Bphi, theta, phi)
        return Bx * (r > rs), By * (r > rs), Bz * (r > rs)

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def Calc2D(r, theta, phi, a, num, log_kmin, log_kmax, A, alpha, delta, rs):
        q = 8 / 3
        p = 0
        gamma = 3

        cospsi = 1. / np.sqrt(1 + (r * np.sin(theta) / a) ** 2)
        sinpsi = (r * np.sin(theta) / a) / np.sqrt(1 + (r * np.sin(theta) / a) ** 2)

        k = np.logspace(log_kmin, log_kmax, num)[:, np.newaxis]
        dk = k * (10 ** ((log_kmax - log_kmin) / (num - 1)) - 1)

        lam = 0.04 * (r / rs) ** 0.8 * rs
        B = A * r ** (-gamma / 2)
        brk = lam * k / r
        denom = (1 + brk ** (p + q))
        numer = dk * k ** (p + 1)
        spectrum = np.sqrt(2 * np.pi * numer) / np.sqrt(denom)
        deltaB = B * spectrum

        dspectrum = (-(0.04 * np.sqrt(np.pi / 2) * (p + q) * numer * brk ** (p + q - 1)) /
                     (spectrum / np.sqrt(2 * np.pi) * denom ** 2) *
                     (0.8 * k / (r * (r / rs) ** 0.2) - brk / (0.04 * r)))

        n = np.trunc(np.sin(alpha) * k)
        alpha = np.real(np.arcsin(n / k) * (np.cos(alpha) > 0) + (np.pi - np.arcsin(n / k)) * (np.cos(alpha) < 0))

        phase = k * (r / a + phi + theta * np.cos(alpha)) + delta

        Br = (-deltaB / r * sinpsi * (np.cos(alpha) * np.cos(phase) +
                                      (1 / np.tan(theta) * np.sin(phase)) / k))

        Btheta = (deltaB * (sinpsi * np.cos(phase) / a -
                            gamma * sinpsi * np.sin(phase) / (2 * r * k) +
                            1 / np.sin(theta) * (cospsi * np.cos(phase) +
                                                 np.sin(theta) * sinpsi * np.sin(phase) / k)) +
                  B * sinpsi * np.sin(phase) * dspectrum / k)

        Bphi = -deltaB / r * np.cos(alpha) * cospsi * np.cos(phase)

        Bx = np.sum(Br, axis=0)
        By = np.sum(Btheta, axis=0)
        Bz = np.sum(Bphi, axis=0)

        Bx, By, Bz = transformations.Sphere2Cart(Bx, By, Bz, theta, phi)
        return Bx * (r > rs), By * (r > rs), Bz * (r > rs)

    def __str__(self):
        s = f"""Parker
        magnitude: {self.magnitude}
        CIR: {self.use_cir}
        Polarization: {self.polarization}
        Noise: {self.use_noise}"""

        if self.use_noise:
            s += f"""
            Min wave length: {self.log_kmin}
            Max wave length: {self.log_kmax}
            Number of waves: {self.noise_num}"""

        return s
