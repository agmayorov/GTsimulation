import datetime
import os
from enum import Enum

import numpy as np

from MagneticFields import AbsBfield, Regions, Units
from MagneticFields.Magnetosphere.Functions.gauss import LoadGaussCoeffs


class GaussModels(Enum):
    IGRF = 1
    CHAOS = 2
    CM = 3
    COV_OBS = 4
    LCS = 5
    SIFM = 6
    DIFI = 7


class GaussTypes(Enum):
    core = 1
    static = 2
    ionosphere = 3


versions_dict = {GaussModels.IGRF: [13],
                 GaussModels.CHAOS: [7.13],
                 GaussModels.CM: [6],
                 GaussModels.COV_OBS: [2],
                 GaussModels.LCS: [1],
                 GaussModels.DIFI: [6],
                 GaussModels.SIFM: [None]}


class Gauss(AbsBfield):
    ToMeters = Units.km2m

    def __init__(self, date: datetime.datetime, model: GaussModels | str, model_type: GaussTypes | str, version=None,
                 coord: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Magnetosphere
        self.Model = model if isinstance(model_type, GaussModels) else GaussModels[model]
        self.type = model_type if isinstance(model_type, GaussTypes) else GaussTypes[model_type]
        self.version = version
        self.Date = date
        self.coord = coord
        self.txt_file_loc = ""
        self.mat_file_loc = ""
        self.npy_file_loc = ""
        self.SetFullModelName()
        self.g, self.h, self.gh = LoadGaussCoeffs(self.npy_file_loc, self.Date)

    def CalcBfield(self, x, y, z, **kwargs):
        altitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi = np.arctan2(y, x)
        theta = np.arccos(z / altitude)

        Rearth_km = 6371.2

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        if self.coord == 0:
            a = 6378.137
            f = 1 / 298.257223563
            b = a * (1 - f)

            rho = np.hypot(a * sintheta, b * costheta)
            r = np.sqrt(
                altitude ** 2 + 2 * altitude * rho + (a ** 4 * sintheta ** 2 + b ** 4 * costheta ** 2) / (rho ** 2))
            cd = (altitude + rho) / r
            sd = (a ** 2 - b ** 2) / rho * costheta * sintheta / r
            oldcos = costheta
            costheta = costheta * cd - sintheta * sd
            sintheta = sintheta * cd + oldcos * sd
        else:
            r = altitude
            cd = 1
            sd = 0

        nmax = np.sqrt(len(self.gh) + 1) - 1

        cosphi = np.cos(np.arange(1, nmax + 1) * phi)
        sinphi = np.sin(np.arange(1, nmax + 1) * phi)

        Pmax = int((nmax + 1) * (nmax + 2) / 2)

        Br = 0
        Bt = 0
        Bp = 0

        P = np.zeros(Pmax)
        P[0] = 1
        P[2] = sintheta

        dP = np.zeros(Pmax)
        dP[0] = 0
        dP[2] = costheta

        m = 1
        n = 0
        coefindex = 0

        a_r = (Rearth_km / r) ** 2

        for Pindex in range(1, Pmax):
            if n < m:
                m = 0
                n += 1
                a_r *= (Rearth_km / r)

            if m < n and Pindex != 2:
                last1n = Pindex - n - 1
                last2n = Pindex - 2 * n + 1
                P[Pindex] = (2 * n - 1) / np.sqrt(n ** 2 - m ** 2) * P[last1n] - np.sqrt(
                    ((n - 1) ** 2 - m ** 2) / (n ** 2 - m ** 2)) * P[last2n]
                dP[Pindex] = (2 * n - 1) / np.sqrt(n ** 2 - m ** 2) * (
                        costheta * dP[last1n] - sintheta * P[last1n]) - np.sqrt(
                    ((n - 1) ** 2 - m ** 2) / (n ** 2 - m ** 2)) * dP[last2n]
            elif Pindex == 2:
                lastn = Pindex - n - 1
                P[Pindex] = np.sqrt(1 - 1 / (2 * m)) * sintheta * P[lastn]
                dP[Pindex] = np.sqrt(1 - 1 / (2 * m)) * (sintheta * dP[lastn] + costheta * P[lastn])

            if m == 0:
                coef = a_r * self.gh[coefindex]
                Br += (n + 1) * coef * P[Pindex]
                Bt -= coef * dP[Pindex]
                coefindex += 1
            else:
                coef = a_r * (self.gh[coefindex] * cosphi[m - 1]) + self.gh[coefindex + 1] * sinphi[m - 1]
                Br += (n + 1) * coef * P[Pindex]
                Bt -= coef * dP[Pindex]

                # TODO: generalize this function, to have vector coordinate inputs and vector magentic field outputs
                if sintheta == 0:
                    Bp -= costheta * a_r * (-self.gh[coefindex] * sinphi[m - 1] + self.gh[coefindex] * cosphi[m - 1]) * \
                          dP[Pindex]
                else:
                    Bp -= 1 / sintheta * a_r * (
                            -self.gh[coefindex] * sinphi[m - 1] + self.gh[coefindex] * cosphi[m - 1]) * P[Pindex]

                coefindex += 2
            m += 1

        Bx = -Bt
        By = Bp
        Bz = -Br

        if self.coord == 0:
            Bx_old = Bx
            Bx = Bx * cd + Bz * sd
            Bz = Bz * cd - Bx_old * sd

        return Bx, By, Bz

    def UpdateState(self, new_date):
        self.Date = new_date
        self.g, self.h.self.gh = LoadGaussCoeffs(self.npy_file_loc, self.Date)

    def SetFullModelName(self):
        assert self.version in versions_dict[self.Model]
        txt_file = ""
        mat_file = ""
        npy_file = ""
        loc = os.path.dirname(os.path.realpath(__file__))
        if self.Model == GaussModels.IGRF:
            self.ModelName = self.Model.name + str(self.version)
            assert self.type == GaussTypes.core
            txt_file = self.ModelName.lower() + 'coeffs.txt'
            mat_file = self.ModelName.lower() + 'coeffs.mat'
            npy_file = self.ModelName.lower() + 'coeffs.npy'
        elif self.Model == GaussModels.CHAOS:
            assert self.type != GaussTypes.ionosphere
            txt_file = self.ModelName + "_" + self.type.name + '.shc.txt'
            mat_file = self.ModelName + "_" + self.type.name + '.mat'
            npy_file = self.ModelName + "_" + self.type.name + '.npy'
        elif self.Model == GaussModels.CM:
            self.ModelName = self.Model.name + str(self.version)
            if self.type == GaussTypes.core:
                txt_file = 'MCO_' + self.ModelName + '.shc.txt'
                mat_file = 'MCO_' + self.ModelName + '.mat'
                npy_file = 'MCO_' + self.ModelName + '.npy'
            elif self.type == GaussTypes.static:
                txt_file = 'MLI_' + self.ModelName + '.shc.txt'
                mat_file = 'MLI_' + self.ModelName + '.mat'
                npy_file = 'MLI_' + self.ModelName + '.npy'
            elif self.type == GaussTypes.ionosphere:
                txt_file = 'MIO_' + self.ModelName + '.shc.txt'
                mat_file = 'MIO_' + self.ModelName + '.mat'
                npy_file = 'MIO_' + self.ModelName + '.npy'
        elif self.Model == GaussModels.COV_OBS:
            self.ModelName = self.Model.name + ".x" + str(self.version) + "-int"
            txt_file = self.ModelName + '.shc.txt'
            mat_file = self.ModelName + '.mat'
            npy_file = self.ModelName + '.npy'
        elif self.Model == GaussModels.LCS:
            self.ModelName = self.Model.name + '-' + str(self.version)
            assert self.type == GaussTypes.static
            txt_file = self.ModelName + '.shc.txt'
            mat_file = self.ModelName + '.mat'
            npy_file = self.ModelName + '.npy'
        elif self.Model == GaussModels.DIFI:
            self.ModelName = self.Model.name + str(self.version)
            assert self.type == GaussTypes.ionosphere
            txt_file = self.ModelName + '.txt'
            mat_file = self.ModelName + '.mat'
            npy_file = self.ModelName + '.npy'
        elif self.Model == GaussModels.SIFM:
            self.ModelName = self.Model.name
            assert self.type != GaussTypes.ionosphere
            txt_file = self.ModelName + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"

        self.txt_file_loc = loc + os.sep + self.ModelName + os.sep + txt_file
        self.mat_file_loc = loc + os.sep + self.ModelName + os.sep + mat_file
        self.npy_file_loc = loc + os.sep + self.ModelName + os.sep + npy_file

    def __str__(self):
        s = f"""{self.Model.name}
        Type: {self.type.name}
        Version: {self.version}"""

        return s

