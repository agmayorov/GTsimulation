import datetime
import scipy.io
import numpy as np
from numba import jit

from Global import Units, Regions
from MagneticFields import AbsBfield


class JF12mod(AbsBfield):
    ToMeters = Units.kpc2m

    def __init__(self, use_noise=True, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Galaxy
        self.ModelName = "JF12mod"
        self.Units = "kpc"
        self.use_noise = use_noise

        # Regular field #

        self.b_disk = np.array([0.1, 3.0, -0.9, -0.8, -2.0, -4.2, 0.0, 2.7])
        self.r_arms = np.array([5.1, 6.3, 7.1, 8.3, 9.8, 11.4, 12.7, 15.5])
        self.b_ring = 0.1
        self.h_disk = 0.4
        self.w_disk = 0.27

        self.pitch = 11.5 * np.pi / 180
        self.sinPitch = np.sin(self.pitch)
        self.cosPitch = np.cos(self.pitch)

        # Toroidal halo
        self.B_n = 1.4
        self.B_s = -1.1
        self.r_n = 9.22
        self.r_s = 16.7
        self.w_h = 0.2
        self.z_0 = 5.3

        # X halo
        self.bX = 4.6
        self.thetaX0 = 49.0 * np.pi / 180
        self.sinThetaX0 = np.sin(self.thetaX0)
        self.cosThetaX0 = np.cos(self.thetaX0)
        self.tanThetaX0 = np.tan(self.thetaX0)
        self.rXc = 4.8
        self.rX = 2.9

        # Non-regular field #

        self.f_a = 0.6
        self.f_i = 0.3

        # Striated field parameter
        self.beta_str = 1.36

        # Disk
        self.b_int_turb = 7.63
        self.b_disk_turb = np.array([10.81, 6.96, 9.59, 6.96, 1.96, 16.34, 37.29, 10.35])
        self.z_disk_turb = 0.61

        # Halo
        self.b_halo_turb = 4.68
        self.r_halo_turb = 10.97
        self.z_halo_turb = 2.84

        # coeffs = np.load(f"Data/G_nCell=250_boxSize=0.5kpc_lMin=4.0pc_lMax=500.0pc_seed=0.npy", allow_pickle=True).item(0)
        coeffs = scipy.io.loadmat(
            "MagneticFields/Galaxy/Data/G_n=512_boxSize=1_lMin=3.9e-03_lMax=1.0e+00_seed=0.mat")
        self.Gx = coeffs["Gx"]
        self.Gy = coeffs["Gy"]
        self.Gz = coeffs["Gz"]

        self.x_grid = coeffs["x_grid"].flatten()
        self.y_grid = coeffs["y_grid"].flatten()
        self.z_grid = coeffs["z_grid"].flatten()

    def CalcBfield(self, x, y, z, **kwargs):
        return self.__calc_b_field(x, y, z,
                                   self.h_disk, self.w_disk,
                                   self.b_ring, self.b_disk,
                                   self.r_arms,
                                   self.pitch, self.sinPitch, self.cosPitch,
                                   self.z_0, self.r_n, self.r_s, self.w_h,
                                   self.B_n, self.B_s,
                                   self.rXc, self.rX,
                                   self.tanThetaX0, self.sinThetaX0, self.cosThetaX0,
                                   self.bX,
                                   self.x_grid, self.y_grid, self.z_grid,
                                   self.Gx, self.Gy, self.Gz,
                                   self.beta_str,
                                   self.b_int_turb, self.b_disk_turb, self.z_disk_turb,
                                   self.b_halo_turb, self.r_halo_turb, self.z_halo_turb,
                                   self.f_a, self.f_i,
                                   self.use_noise)

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def __calc_b_field(x, y, z,
                       h_disk, w_disk,
                       b_ring, b_disk, r_arms,
                       pitch, sinPitch, cosPitch,
                       z_0, r_n, r_s, w_h,
                       B_n, B_s,
                       rXc, rX,
                       tanThetaX0, sinThetaX0, cosThetaX0,
                       bX,
                       x_grid, y_grid, z_grid,
                       Gx, Gy, Gz,
                       beta_str,
                       b_int_turb, b_disk_turb, z_disk_turb,
                       b_halo_turb, r_halo_turb, z_halo_turb,
                       f_a, f_i,
                       use_noise):
        R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        lfDisk = 1 / (1 + np.exp(-2 * (np.abs(z) - h_disk) / w_disk))

        # Disk component
        Bx_d = 0
        By_d = 0
        disk_idx = 0
        if 3 <= r < 5:
            # Molecular ring
            bMag = b_ring * (5 / r) * (1 - lfDisk)
            Bx_d = -bMag * np.sin(phi)
            By_d = bMag * np.cos(phi)
        elif 5 <= r < 20:
            plus2pi = 0
            r_negx = r * np.exp(-(phi - np.pi + plus2pi) / np.tan(np.pi / 2 - pitch))
            while r_negx > r_arms[7]:
                plus2pi += 2 * np.pi
                r_negx = r * np.exp(-(phi - np.pi + plus2pi) / np.tan(np.pi / 2 - pitch))

            disk_idx = np.where(r_negx <= r_arms)[0][0]
            bMag = b_disk[disk_idx]
            bMag *= (5 / r) * (1 - lfDisk)
            Bx_d = bMag * (sinPitch * np.cos(phi) - cosPitch * np.sin(phi))
            By_d = bMag * (sinPitch * np.sin(phi) + cosPitch * np.cos(phi))

        # Toroidal component
        Bx_t = 0
        By_t = 0
        if R >= 1 and r <= 20:
            bMagH = np.exp(-np.abs(z) / z_0) * lfDisk
            if z >= 0:
                lf_north = 1 / (1 + np.exp(-2 * (np.abs(r) - r_n) / w_h))
                bMagH *= B_n * (1 - lf_north)
            else:
                lf_south = 1 / (1 + np.exp(-2 * (np.abs(r) - r_s) / w_h))
                bMagH *= B_s * (1 - lf_south)

            Bx_t = -bMagH * np.sin(phi)
            By_t = bMagH * np.cos(phi)

        # X-field component
        Bx_x = 0
        By_x = 0
        Bz_x = 0
        if R >= 1 and r <= 20:
            rc = rXc + np.abs(z) / tanThetaX0
            if 0 < r < rc:
                rp = r * rXc / rc
                bMagX = bX * np.exp(-rp / rX) * (rp / r) ** 2
                thetaX = np.pi / 2 if z == 0 else np.arctan(np.abs(z) / (r - rp))
                sinThetaX = np.sin(thetaX)
                cosThetaX = np.cos(thetaX)
            elif r == 0:
                bMagX = bX * (rXc / rc) ** 2
                thetaX = np.pi / 2
                sinThetaX = np.sin(thetaX)
                cosThetaX = np.cos(thetaX)
            else:
                rp = r - np.abs(z) / tanThetaX0
                bMagX = bX * np.exp(-rp / rX) * (rp / r)
                sinThetaX = sinThetaX0
                cosThetaX = cosThetaX0
            Bx_x = np.sign(z) * bMagX * cosThetaX * np.cos(phi)
            By_x = np.sign(z) * bMagX * cosThetaX * np.sin(phi)
            Bz_x = bMagX * sinThetaX

            # Total REGULAR field

        Bx = Bx_x + Bx_d + Bx_t
        By = By_x + By_d + By_t
        Bz = Bz_x
        Babs = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

        Bx_aniso = 0
        By_aniso = 0
        Bz_aniso = 0

        Bx_iso = 0
        By_iso = 0
        Bz_iso = 0

        if use_noise:
            if r < 20 and np.abs(z) < 20:
                ix = np.where(x % x_grid[-1] >= x_grid[:-1])[0][-1]
                iy = np.where(y % y_grid[-1] >= y_grid[:-1])[0][-1]
                iz = np.where(z % z_grid[-1] >= z_grid[:-1])[0][-1]
                Gx_p = Gx[ix, iy, iz]
                Gy_p = Gy[ix, iy, iz]
                Gz_p = Gz[ix, iy, iz]

                # Anisotropic random field
                B_aniso_rms = np.sqrt(1.5 * beta_str) * Babs * (Bx * Gx_p + By * Gy_p + Bz * Gz_p)
                Bx_aniso = B_aniso_rms * Bx
                By_aniso = B_aniso_rms * By
                Bz_aniso = B_aniso_rms * Bz

                # Isotropic random field
                if r < 5:
                    B_iso_disk = b_int_turb
                else:
                    B_iso_disk = b_disk_turb[disk_idx]
                    B_iso_disk = B_iso_disk * (5 / r)
                B_iso_disk *= np.exp(-0.5 * (z/z_disk_turb)**2)

                # Halo
                B_iso_halo = b_halo_turb * np.exp(-r/r_halo_turb) * np.exp(-0.5 * (z/z_halo_turb)**2)

                # General
                B_iso_rms = np.sqrt(B_iso_halo**2  + B_iso_disk**2)
                Bx_iso = B_iso_rms * Gx_p
                By_iso = B_iso_rms * Gy_p
                Bz_iso = B_iso_rms * Gz_p

        return 0.1 * (Bx + f_a*Bx_aniso + f_i * Bx_iso), 0.1 * (By + f_a*By_aniso + f_i * By_iso), 0.1 * (Bz + f_a*Bz_aniso + f_i * Bz_iso)

    def UpdateState(self, new_date):
        pass

    def __str__(self):
        s = f"""JF12mod
        Noise: {self.use_noise}"""

        return s
