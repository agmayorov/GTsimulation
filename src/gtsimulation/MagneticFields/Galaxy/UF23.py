import datetime
from typing import Callable, Any

import scipy.io
import numpy as np
from numba import jit
from enum import Enum

from Global import Units, Regions
from MagneticFields import AbsBfield
import os
import sys


class ModelTypes(Enum):
    base = 1,
    expX = 2,
    neCL = 3,
    twistX = 4,
    nebCor = 5,
    cre10 = 6,
    synCG = 7


class UF23(AbsBfield):
    ToMeters = Units.kpc2m

    def __init__(self, model_type: ModelTypes | str = ModelTypes.base, use_noise=True, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Galaxy
        self.ModelName = "UF23"
        self.model_type = model_type if isinstance(model_type, ModelTypes) else ModelTypes[model_type]
        self.Units = "kpc"
        self.use_noise = use_noise
        self.float_min = sys.float_info.min

        # Regular field #
        # Disk field
        self.r_0 = 5  # kpc
        self.r_1 = 5  # kpc
        self.r_2 = 20  # kpc
        self.w_1 = 0.5  # kpc
        self.w_2 = 0.5  # kpc

        # TODO add 'twistX' model
        match self.model_type:
            case ModelTypes.base:
                # Disk field
                self.pitch = 10.11 * np.pi / 180  # rad
                self.z_disk = 0.794  # kpc
                self.w_disk = 0.107  # kpc
                self.B_1 = 1.09  # muG
                self.B_2 = 2.66  # muG
                self.B_3 = 3.12  # muG
                self.phi_1 = 263 * np.pi / 180  # rad
                self.phi_2 = 97.8 * np.pi / 180  # rad
                self.phi_3 = 35.1 * np.pi / 180  # rad

                # Toroidal halo
                self.B_n = 3.26  # muG
                self.B_s = -3.09  # muG
                self.z_t = 4.0  # kpc
                self.r_t = 10.19  # kpc
                self.w_t = 1.7  # kpc
                self.t = None  # Myr

                # Poloidal halo
                self.B_p = 0.978  # muG
                self.p = 1.43
                self.z_p = 4.5  # kpc
                self.r_p = 7.29  # kpc
                self.w_p = 0.112  # kpc
                self.a_c = 1e6  # kpc -- because 'a_c' -> inf

                # Other model parameters
                self.xi = 0.346
                self.beta_str = 1 - (1 + self.xi) ** 2

            case ModelTypes.expX:
                # Disk field
                self.pitch = 10.03 * np.pi / 180  # rad
                self.z_disk = 0.715  # kpc
                self.w_disk = 0.099  # kpc
                self.B_1 = 0.99  # muG
                self.B_2 = 2.18  # muG
                self.B_3 = 3.12  # muG
                self.phi_1 = 247 * np.pi / 180  # rad
                self.phi_2 = 98.6 * np.pi / 180  # rad
                self.phi_3 = 34.9 * np.pi / 180  # rad

                # Toroidal halo
                self.B_n = 2.71  # muG
                self.B_s = -2.57  # muG
                self.z_t = 5.5  # kpc
                self.r_t = 10.13  # kpc
                self.w_t = 2.1  # kpc
                self.t = None  # Myr

                # Poloidal halo
                self.B_p = 5.8  # muG
                self.p = 1.95
                self.z_p = 2.37  # kpc
                self.r_p = 7.29  # kpc
                self.w_p = None  # kpc
                self.a_c = 6.2  # kpc

                # Other model parameters
                self.xi = 0.51
                self.beta_str = 1 - (1 + self.xi) ** 2

            case ModelTypes.neCL:
                # Disk field
                self.pitch = 11.9 * np.pi / 180  # rad
                self.z_disk = 0.674  # kpc
                self.w_disk = 0.061  # kpc
                self.B_1 = 1.43  # muG
                self.B_2 = 1.4  # muG
                self.B_3 = 3.44  # muG
                self.phi_1 = 200 * np.pi / 180  # rad
                self.phi_2 = 135 * np.pi / 180  # rad
                self.phi_3 = 65 * np.pi / 180  # rad

                # Toroidal halo
                self.B_n = 2.63  # muG
                self.B_s = -2.57  # muG
                self.z_t = 4.6  # kpc
                self.r_t = 10.13  # kpc
                self.w_t = 1.15  # kpc
                self.t = None  # Myr

                # Poloidal halo
                self.B_p = 0.984  # muG
                self.p = 1.68
                self.z_p = 3.65  # kpc
                self.r_p = 7.41  # kpc
                self.w_p = 0.142  # kpc
                self.a_c = 1e6  # kpc -- because 'a_c' -> inf

                # Other model parameters
                self.xi = 0.336
                self.beta_str = 1 - (1 + self.xi) ** 2

            case ModelTypes.nebCor:
                # Disk field
                self.pitch = 10.15 * np.pi / 180  # rad
                self.z_disk = 0.812  # kpc
                self.w_disk = 0.119  # kpc
                self.B_1 = 1.41  # muG
                self.B_2 = 3.53  # muG
                self.B_3 = 4.13  # muG
                self.phi_1 = 264 * np.pi / 180  # rad
                self.phi_2 = 97.6 * np.pi / 180  # rad
                self.phi_3 = 36.4 * np.pi / 180  # rad

                # Toroidal halo
                self.B_n = 4.6  # muG
                self.B_s = -4.5  # muG
                self.z_t = 3.6  # kpc
                self.r_t = 10.21  # kpc
                self.w_t = 1.7  # kpc
                self.t = None  # Myr

                # Poloidal halo
                self.B_p = 1.35  # muG
                self.p = 1.34
                self.z_p = 4.8  # kpc
                self.r_p = 7.25  # kpc
                self.w_p = 0.143  # kpc
                self.a_c = 1e6  # kpc -- because 'a_c' -> inf

                # Other model parameters
                self.xi = 0.0
                self.beta_str = 1 - (1 + self.xi) ** 2

            case ModelTypes.synCG:
                # Disk field
                self.pitch = 9.90 * np.pi / 180  # rad
                self.z_disk = 0.622  # kpc
                self.w_disk = 0.067  # kpc
                self.B_1 = 0.81  # muG
                self.B_2 = 2.06  # muG
                self.B_3 = 2.94  # muG
                self.phi_1 = 230 * np.pi / 180  # rad
                self.phi_2 = 97.4 * np.pi / 180  # rad
                self.phi_3 = 32.9 * np.pi / 180  # rad

                # Toroidal halo
                self.B_n = 2.40  # muG
                self.B_s = -2.09  # muG
                self.z_t = 5.6  # kpc
                self.r_t = 9.42  # kpc
                self.w_t = 0.92  # kpc
                self.t = None  # Myr

                # Poloidal halo
                self.B_p = 0.809  # muG
                self.p = 1.58
                self.z_p = 3.53  # kpc
                self.r_p = 7.46  # kpc
                self.w_p = 0.150  # kpc
                self.a_c = 1e6  # kpc -- because 'a_c' -> inf

                # Other model parameters
                self.xi = 0.63
                self.beta_str = 1 - (1 + self.xi) ** 2

            case ModelTypes.cre10:
                # Disk field
                self.pitch = 10.16 * np.pi / 180  # rad
                self.z_disk = 0.808  # kpc
                self.w_disk = 0.108  # kpc
                self.B_1 = 1.20  # muG
                self.B_2 = 2.75  # muG
                self.B_3 = 3.21  # muG
                self.phi_1 = 265 * np.pi / 180  # rad
                self.phi_2 = 98.2 * np.pi / 180  # rad
                self.phi_3 = 35.9 * np.pi / 180  # rad

                # Toroidal halo
                self.B_n = 3.7  # muG
                self.B_s = -3.50  # muG
                self.z_t = 2.9  # kpc
                self.r_t = 10.41  # kpc
                self.w_t = 1.7  # kpc
                self.t = None  # Myr

                # Poloidal halo
                self.B_p = 0.969  # muG
                self.p = 1.42
                self.z_p = 4.6  # kpc
                self.r_p = 7.30  # kpc
                self.w_p = 0.109  # kpc
                self.a_c = 1e6  # kpc -- because 'a_c' -> inf

                # Other model parameters
                self.xi = 0.250
                self.beta_str = 1 - (1 + self.xi) ** 2

    def CalcBfield(self, x, y, z, **kwargs):
        return self.__calc_b_field(x, y, z,
                                   self.pitch, self.z_disk, self.w_disk,
                                   self.B_1, self.B_2, self.B_3,
                                   self.phi_1, self.phi_2, self.phi_3,
                                   self.r_0, self.r_1, self.r_2,
                                   self.w_1, self.w_2,
                                   self.B_n, self.B_s,
                                   self.z_t, self.r_t, self.w_t, self.t,
                                   self.B_p, self.p, self.z_p, self.r_p, self.w_p, self.a_c,
                                   self.float_min,
                                   self.use_noise, self.model_type.value[0])

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def __calc_b_field(x, y, z,
                       pitch, z_disk, w_disk,
                       B_1, B_2, B_3,
                       phi_1, phi_2, phi_3,
                       r_0, r_1, r_2,
                       w_1, w_2,
                       B_n, B_s,
                       z_t, r_t, w_t, t,
                       B_p, p, z_p, r_p, w_p, a_c,
                       float_min,
                       use_noise, model_type_value):

        R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        phi = phi if phi >= 0 else phi + 2 * np.pi

        # Disk
        phi_0 = phi - np.log(r / r_0) / np.tan(pitch)
        B_ref = (B_1 * np.cos(1 * (phi_0 - phi_1)) + B_2 * np.cos(2 * (phi_0 - phi_2)) +
                 B_3 * np.cos(3 * (phi_0 - phi_3)))

        sigmoid = lambda x_arg: 1.0 / (1.0 + np.exp(-x_arg))

        h_disk = lambda z_arg: 1 - sigmoid((np.abs(z_arg) - z_disk) / w_disk)
        g_disk = lambda r_arg: ((1 - sigmoid((r_arg - r_2) / w_2)) * sigmoid((r_arg - r_1) / w_1) *
                                (1 - np.exp(-r_arg ** 2)))

        Br_d = np.sin(pitch) * r_0 / r * B_ref * h_disk(z) * g_disk(r)
        Bphi_d = np.cos(pitch) * r_0 / r * B_ref * h_disk(z) * g_disk(r)

        Bx_d = Br_d * np.cos(phi) - Bphi_d * np.sin(phi)
        By_d = Br_d * np.sin(phi) + Bphi_d * np.cos(phi)

        # Toroidal halo
        Bphi_t = (1 - h_disk(z)) * np.exp(-np.abs(z) / z_t) * (1 - sigmoid((r - r_t) / w_t))
        if z > 0:
            Bphi_t *= B_n
        else:
            Bphi_t *= B_s
        Bx_t = Bphi_t * np.sin(phi)
        By_t = Bphi_t * np.cos(phi)

        # Poloidal halo
        c = np.power(a_c / z_p, p)
        delta = np.power(a_c, p) + c * np.power(np.abs(z), p) - np.power(r, p)
        k = 4 * np.power(a_c, p) * np.power(r, p)
        a_p = 0.5 * k / (np.sqrt(np.power(delta, 2) + k) + delta)
        a = np.power(a_p, 1 / p)
        r_over_a = 1 / np.power(2 * np.power(a_c, p) / (np.sqrt(np.power(delta, 2) + k) + delta), 1 / p)
        # sign_z = -1 if z < 0 else 1

        # Radial functions
        if model_type_value == 2:
            f_x = np.exp(-a / r_p)
        else:
            f_x = 1 - sigmoid((a - r_p) / w_p)
        B0 = B_p * f_x

        if r <= float_min:
            Br_p = 0.0
        else:
            Br_p = B0 * c * a / r_over_a * np.sign(z) * np.power(np.abs(z), p - 1) / np.sqrt(np.power(delta, 2) + k)
        Bz_p = B0 * np.power(r_over_a, p - 2) * (a_p + np.power(a_c, p)) / np.sqrt(np.power(delta, 2) + k)

        Bx_p = Br_p * np.cos(phi)
        By_p = -Br_p * np.sin(phi)

        # Total regular field
        Bx = Bx_d + Bx_t + Bx_p
        By = By_d + By_t + By_p
        Bz = Bz_p
        Babs = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

        # TODO add irregular field

        return 0.1 * Bx, 0.1 * By, 0.1 * Bz

    def UpdateState(self, new_date):
        pass

    def to_string(self):
        s = f"""UF23
        Noise: {self.use_noise}
        Model type: {self.model_type.name}"""

        return s
