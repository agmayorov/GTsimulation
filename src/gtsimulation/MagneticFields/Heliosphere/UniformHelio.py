import numpy as np
from numba import jit

from gtsimulation.Global import Units, Regions
from gtsimulation.MagneticFields import AbsBfield


class UniformHelio(AbsBfield):
    ToMeters = Units.AU2m
    def __init__(self, use_noise=True, use_reg=True, use_slab=True, use_2d=True, coeff_slab = 0.4, coeff_2d = 39000, q=1, nu=5/3, p=2.61, **kwargs):
        super().__init__(**kwargs)

        self.Region = Regions.Heliosphere
        self.ModelName = "UniformHelio"
        self.Units = "AU"

        self.B0 = 6
        self.use_reg = use_reg
        self.use_noise = use_noise
        self.use_2d = use_2d
        self.use_slab = use_slab

        self.coeff_slab = coeff_slab
        self.coeff_2d = coeff_2d

        self.q = q  # 2d energy spectral index
        self.nu = nu  # slab/2d inertial range spectral index
        self.p = p  # Dissipation range spectral index

        self.l1_slab = 4.5 * 1e6 / Units.AU2km  # Slab inertial range onset length scale
        self.l1_2d = 1e6 / Units.AU2km  # 2d inertial range onset length scale

        self.l2_slab = 1.98 * 1e3 / Units.AU2km  # slab dissipation range onset length scale
        self.l2_2d = 3.6 * 1e4 / Units.AU2km  # 2d dissipation range onset length scale

        self.l0_slab = 3.51 * 1e7 / Units.AU2km  # slab largest length scale
        self.l0_2d = 1.38 * 1e7 / Units.AU2km  # 2d largest length scale

        self.l3_slab = 300 / Units.AU2km  # slab smallest length scale
        self.l3_2d = self.l2_2d  # 2d smallest length scale

        self.Nz = 2 ** 19
        self.Nx = self.Ny = 2 ** 12

        self.Bx_slab, self.By_slab = np.zeros(self.Nz), np.zeros(self.Nz)
        self.Bx_2d, self.By_2d = np.zeros((self.Nx, self.Ny)), np.zeros((self.Nx, self.Ny))

        if self.use_noise:
            if self.use_slab:
                self.Bx_slab, self.By_slab = self._generate_slab()
            if self.use_2d:
                self.Bx_2d, self.By_2d = self._generate_2d()

    def _generate_slab(self):
        N = self.Nz
        # k_max = 2 * np.pi/self.l3_slab
        # k_min = 2 * np.pi/self.l0_slab
        z_max = self.l0_slab
        k = 2 * np.pi / z_max * np.arange(N) + 1e-12
        # k = np.linspace(k_min, k_max, N)
        P_m = self.calc_spectrum_slab(k)
        h = z_max / N
        X_m = np.sqrt(N * P_m / (2 * h)) * np.exp(1.j * np.random.rand(N) * 2 * np.pi)
        Y_m = np.sqrt(N * P_m / (2 * h)) * np.exp(1.j * np.random.rand(N) * 2 * np.pi)

        Bx = np.real(np.fft.ifft(X_m))
        By = np.real(np.fft.ifft(Y_m))

        return self.coeff_slab*Bx, self.coeff_slab*By

    def calc_spectrum_slab(self, k_par):
        return self._calc_spectrum_slab(k_par, self.nu, self.p, self.l0_slab, self.l1_slab, self.l2_slab, self.l3_slab)

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def _calc_spectrum_slab(k, s, p, l0, l1, l2, l3):
        k0 = 2 * np.pi / l0
        k1 = 2 * np.pi / l1
        k2 = 2 * np.pi / l2
        k3 = 2 * np.pi / l3

        return 1 * (k0 <= k) * (k < k1) + \
            (l1 * k / (2 * np.pi)) ** (-s) * (k1 <= k) * (k < k2) + \
            (l1 / l2) ** (-s) * (l2 * k / (2 * np.pi)) ** (-p) * (k2 <= k) * (k <= k3)

    def _generate_2d(self):
        N = self.Ny
        x_max = y_max = self.l0_2d
        kx = 2 * np.pi / x_max * np.arange(N) + 1e-12
        ky = 2 * np.pi / y_max * np.arange(N) + 1e-12
        k = np.linalg.norm(np.array(np.meshgrid(kx, ky)), axis=0)
        kx, ky = np.meshgrid(kx, ky)
        P_m = self.calc_spectrum_2d(k)
        A_m = P_m / k ** 3

        h = x_max / N
        Z_m = np.sqrt(N * A_m / (2 * h)) * np.exp(1.j * np.random.rand(N) * 2 * np.pi) * \
              np.exp(1.j * np.random.rand(N)[np.newaxis].T * 2 * np.pi)

        X_m = ky * Z_m
        Y_m = -kx * Z_m

        Bx = np.real(np.fft.ifftn(X_m))
        By = np.real(np.fft.ifftn(Y_m))

        return self.coeff_2d*Bx, self.coeff_2d*By  # in order for anisotropy (2d/slab) to be 80%/20%

    def calc_spectrum_2d(self, k_perp):
        return self._calc_spectrum_2d(k_perp, self.nu, self.q, self.l0_2d, self.l1_2d, self.l3_2d)

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def _calc_spectrum_2d(k, s, q, l0, l1, l3):
        k0 = 2 * np.pi / l0
        k1 = 2 * np.pi / l1
        k3 = 2 * np.pi / l3

        return (l0 * k / (2 * np.pi)) ** (-q) * (k0 <= k) * (k < k1) + \
            (l0 / l1) ** (-q) * (l1 * k / (2 * np.pi)) ** (-s) * (k1 <= k) * (k <= k3)

    def UpdateState(self, new_date):
        pass

    def CalcBfield(self, x, y, z, **kwargs):
        Bx = By = Bz = 0
        if self.use_reg:
            Bx, By, Bz = 0, 0, self.B0
        if not self.use_noise:
            return Bx, By, Bz

        Bx_n, By_n = self.__calc_noise(self.use_slab, self.Bx_slab, self.By_slab, self.Nz,
                                       self.use_2d, self.Bx_2d, self.By_2d, self.Nx, self.Ny,
                                       self.l0_2d, self.l0_2d, self.l0_slab,
                                       x, y, z)

        return Bx + Bx_n, By + By_n, Bz

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __calc_noise(use_slab, Bx_slab, By_slab, Nz,
                     use_2d, Bx_2d, By_2d, Nx, Ny,
                     x_max, y_max, z_max,
                     x, y, z):
        Bx, By = 0, 0
        if use_slab:
            rz = z % z_max
            iz = rz / z_max * Nz
            iz_f = int(rz / z_max * Nz)
            iz_c = iz_f + 1
            Bx += Bx_slab[iz_f] * (iz_c - iz) + Bx_slab[iz_c if iz_c != Nz else 0] * (iz - iz_f)
            By += By_slab[iz_f] * (iz_c - iz) + By_slab[iz_c if iz_c != Nz else 0] * (iz - iz_f)

        if use_2d:
            rx = x % x_max
            ry = y % y_max
            ix = rx / x_max * Nx
            ix_f = int(rx / x_max * Nx)
            ix_c = ix_f + 1
            iy = ry / y_max * Ny
            iy_f = int(ry / y_max * Ny)
            iy_c = iy_f + 1
            Bx_xf = Bx_2d[ix_f, iy_f] * (ix_c - ix) + Bx_2d[ix_c if ix_c != Nx else 0, iy_f] * (ix - ix_f)
            Bx_xc = Bx_2d[ix_f, iy_c if iy_c != Ny else 0] * (ix_c - ix) + Bx_2d[
                ix_c if ix_c != Nx else 0, iy_c if iy_c != Ny else 0] * (ix - ix_f)
            Bx += (iy_c - iy) * Bx_xf + (iy - iy_f) * Bx_xc

            By_xf = By_2d[ix_f, iy_f] * (ix_c - ix) + By_2d[ix_c if ix_c != Nx else 0, iy_f] * (ix - ix_f)
            By_xc = By_2d[ix_f, iy_c if iy_c != Ny else 0] * (ix_c - ix) + By_2d[
                ix_c if ix_c != Nx else 0, iy_c if iy_c != Ny else 0] * (ix - ix_f)
            By += (iy_c - iy) * By_xf + (iy - iy_f) * By_xc
        return Bx, By

    def to_string(self):
        s = f"""{self.ModelName}
            Regular: {self.use_reg}
            Noise: {self.use_noise}
            """

        if self.use_noise:
            s += f"""
            Using Slab: {self.use_slab}
            Coeff Slab: {self.coeff_slab}
            Using 2D: {self.use_2d}
            Coeff 2D: {self.coeff_2d}
	    q: {self.q}
	    nu: {self.nu}
	    p: {self.p}"""

        return s
