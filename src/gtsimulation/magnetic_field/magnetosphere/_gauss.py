import datetime
import os
from enum import Enum
from typing import Union

import numpy as np
from numba import jit

from gtsimulation.common import Units, Regions
from gtsimulation.magnetic_field import AbsBfield
from gtsimulation.magnetic_field.magnetosphere.Functions.gauss import LoadGaussCoeffs


class GaussPlanets(Enum):
    Earth = 1
    Jupiter = 2
    Saturn = 3
    Uranus = 4
    Neptune = 5
    Mercury = 6
    Mars = 7


class GaussMoons(Enum):
    Ganymede = 1


GaussBodies = Union[GaussPlanets, GaussMoons]


class GaussModels(Enum):
    IGRF = 1            #Planets
    CHAOS = 2
    CM = 3
    COV_OBS = 4
    LCS = 5
    SIFM = 6
    DIFI = 7
    JRM33 = 8
    JRM09 = 9
    Cassini11 = 10
    Cassini11plus = 11
    Q3 = 12
    O8 = 13
    MBF_a_n = 14
    Langlais2019 = 15
    MagModel4 = 16       #Moon


class GaussTypes(Enum):
    core = 1
    static = 2
    ionosphere = 3


radius_dict = {GaussPlanets.Earth: 6371.2,
               GaussPlanets.Jupiter: 71492.0,
               GaussPlanets.Saturn: 60268.0,
               GaussPlanets.Uranus: 25600.0,
               GaussPlanets.Neptune: 24765.0,
               GaussPlanets.Mercury: 2440.0,
               GaussPlanets.Mars: 3393.5,
               GaussMoons.Ganymede: 2631.2}


body_dict = {GaussPlanets.Earth: [
             GaussModels.IGRF,
             GaussModels.CHAOS,
             GaussModels.CM,
             GaussModels.COV_OBS,
             GaussModels.LCS,
             GaussModels.SIFM,
             GaussModels.DIFI,
             ],
             GaussPlanets.Jupiter: [
             GaussModels.JRM33,
             GaussModels.JRM09
             ],
             GaussPlanets.Saturn:  [
             GaussModels.Cassini11,
             GaussModels.Cassini11plus
             ],
             GaussPlanets.Uranus:  [
             GaussModels.Q3
             ],
             GaussPlanets.Neptune: [
             GaussModels.O8
             ],
             GaussPlanets.Mercury: [
             GaussModels.MBF_a_n
             ],
             GaussPlanets.Mars: [
             GaussModels.Langlais2019
             ],
             GaussMoons.Ganymede: [
             GaussModels.MagModel4
             ]}


versions_dict = {GaussModels.IGRF: [13, 14],
                 GaussModels.CHAOS: [7.18, 8.5],
                 GaussModels.CM: [6],
                 GaussModels.COV_OBS: [2],
                 GaussModels.LCS: [1],
                 GaussModels.DIFI: [6],
                 GaussModels.SIFM: [None],
                 GaussModels.JRM33: [None],
                 GaussModels.JRM09: [None],
                 GaussModels.Cassini11: [None],
                 GaussModels.Cassini11plus: [None],
                 GaussModels.Q3: [None],
                 GaussModels.O8: [None],
                 GaussModels.MBF_a_n: [None],
                 GaussModels.Langlais2019: [None],
                 GaussModels.MagModel4: [None]}


class Gauss(AbsBfield):
    ToMeters = Units.km2m

    def __init__(self, date: datetime.datetime, model: GaussModels | str, model_type: GaussTypes | str, version=None,
                 planet: Union[GaussPlanets, str, None] = None,
                 moon: Union[GaussMoons, str, None] = None,
                 coord: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Magnetosphere
        self.Units = "km"
        body = planet or moon or GaussPlanets.Earth
        self.BBody = (GaussPlanets[body] if body in GaussPlanets.__members__ else GaussMoons[body]) if isinstance(body, str) else body
        self.body_type = "Planet" if isinstance(self.BBody, GaussPlanets) else "Moon"
        self.Model = model if isinstance(model, GaussModels) else GaussModels[model]
        self.type = model_type if isinstance(model_type, GaussTypes) else GaussTypes[model_type]
        self.Rbody_km = radius_dict[self.BBody]
        self.version = version
        self.Date = date
        self.coord = coord
        self.txt_file_loc = ""
        self.mat_file_loc = ""
        self.npy_file_loc = ""
        self.SetFullModelName()
        self.g, self.h, self.gh = LoadGaussCoeffs(self.npy_file_loc, self.Date)

    def CalcBfield(self, x, y, z, **kwargs):
        coord = self.coord
        gh = self.gh
        Rbody_km = self.Rbody_km
        Bx, By, Bz = self.__calc_b(x, y, z, coord, gh, Rbody_km)

        return Bx, By, Bz

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __calc_b(x, y, z, coord, gh, Rbody_km):
        altitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi = np.arctan2(y, x)
        theta = np.arccos(z / altitude)
        lat = np.pi/2 - theta
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        # if coord == 0:
        #     # TODO: to fix r calculation
        #     a = 6378.137
        #     f = 1 / 298.257223563
        #     b = a * (1 - f)
        #
        #     rho = np.hypot(a * sintheta, b * costheta)
        #     r = np.sqrt(
        #         altitude ** 2 + 2 * altitude * rho + (a ** 4 * sintheta ** 2 + b ** 4 * costheta ** 2) / (rho ** 2))
        #     cd = (altitude + rho) / r
        #     sd = (a ** 2 - b ** 2) / rho * costheta * sintheta / r
        #     oldcos = costheta
        #     costheta = costheta * cd - sintheta * sd
        #     sintheta = sintheta * cd + oldcos * sd
        # else:
        r = altitude
        cd = 1
        sd = 0
        nmax = np.sqrt(len(gh) + 1) - 1
        cosphi = np.cos(np.arange(1, nmax + 1) * phi)
        sinphi = np.sin(np.arange(1, nmax + 1) * phi)
        Pmax = int((nmax + 1) * (nmax + 2) / 2)
        Br = 0.
        Bt = 0.
        Bp = 0.
        P = np.zeros(Pmax)
        P[0] = 1
        P[2] = sintheta
        dP = np.zeros(Pmax)
        dP[0] = 0
        dP[2] = costheta
        m = 1
        n = 0
        coefindex = 0
        a_r = (Rbody_km / r) ** 2
        for Pindex in range(1, Pmax):
            if n < m:
                m = 0
                n += 1
                a_r *= (Rbody_km / r)

            if m < n and Pindex != 2:
                last1n = Pindex - n
                last2n = Pindex - 2 * n + 1
                P[Pindex] = (2 * n - 1) / np.sqrt(n ** 2 - m ** 2) * costheta * P[last1n] - np.sqrt(
                    ((n - 1) ** 2 - m ** 2) / (n ** 2 - m ** 2)) * P[last2n]
                dP[Pindex] = (2 * n - 1) / np.sqrt(n ** 2 - m ** 2) * (
                        costheta * dP[last1n] - sintheta * P[last1n]) - np.sqrt(
                    ((n - 1) ** 2 - m ** 2) / (n ** 2 - m ** 2)) * dP[last2n]
            elif Pindex != 2:
                lastn = Pindex - n - 1
                P[Pindex] = np.sqrt(1 - 1 / (2 * m)) * sintheta * P[lastn]
                dP[Pindex] = np.sqrt(1 - 1 / (2 * m)) * (sintheta * dP[lastn] + costheta * P[lastn])

            if m == 0:
                coef = a_r * gh[coefindex]
                Br += (n + 1) * coef * P[Pindex]
                Bt -= coef * dP[Pindex]
                coefindex += 1
            else:
                coef = a_r * (gh[coefindex] * cosphi[m - 1] + gh[coefindex + 1] * sinphi[m - 1])
                Br += (n + 1) * coef * P[Pindex]
                Bt -= coef * dP[Pindex]

                if sintheta == 0:
                    Bp -= costheta * a_r * (-gh[coefindex] * sinphi[m - 1] + gh[coefindex+1] * cosphi[m - 1]) * \
                          dP[Pindex]
                else:
                    Bp -= 1 / sintheta * a_r * m * (
                            -gh[coefindex] * sinphi[m - 1] + gh[coefindex+1] * cosphi[m - 1]) * P[Pindex]

                coefindex += 2
            m += 1
        Bx = -Bt
        By = Bp
        Bz = -Br
        # if coord == 0:
        #     Bx_old = Bx
        #     Bx = Bx * cd + Bz * sd
        #     Bz = Bz * cd - Bx_old * sd

        My = np.array([[-np.sin(lat), 0., np.cos(lat)],
                       [0, 1., 0],
                       [-np.cos(lat), 0., -np.sin(lat)]])

        Mz = np.array([[np.cos(phi), np.sin(phi), 0.],
                       [-np.sin(phi), np.cos(phi), 0.],
                       [0., 0., 1.]])

        B = np.array([Bx, By, Bz]) @ My @ Mz

        Bx = B[0]
        By = B[1]
        Bz = B[2]

        return Bx, By, Bz

    def UpdateState(self, new_date):
        self.Date = new_date
        self.g, self.h, self.gh = LoadGaussCoeffs(self.npy_file_loc, self.Date)

    def SetFullModelName(self):
        assert self.Model in body_dict[self.BBody]
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
            self.ModelName = self.Model.name + '-' + str(self.version)
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
        elif self.Model == GaussModels.JRM33:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.core
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"
        elif self.Model == GaussModels.JRM09:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.core
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"
        elif self.Model == GaussModels.Cassini11:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.core
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"
        elif self.Model == GaussModels.Cassini11plus:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.core
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"
        elif self.Model == GaussModels.Q3:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.core
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"
        elif self.Model == GaussModels.O8:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.core
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"
        elif self.Model == GaussModels.MBF_a_n:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.core
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"
        elif self.Model == GaussModels.Langlais2019:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.static
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"
        elif self.Model == GaussModels.MagModel4:
            self.ModelName = self.Model.name
            assert self.type == GaussTypes.core
            txt_file = self.ModelName + "_" + self.type.name + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"

        self.txt_file_loc = loc + os.sep + self.BBody.name + os.sep + self.ModelName + os.sep + txt_file
        self.mat_file_loc = loc + os.sep + self.BBody.name + os.sep + self.ModelName + os.sep + mat_file
        self.npy_file_loc = loc + os.sep + self.BBody.name + os.sep + self.ModelName + os.sep + npy_file

    def to_string(self):
        s = f"""{self.Model.name}
        {self.body_type}: {self.BBody.name}
        Type: {self.type.name}
        Version: {self.version}"""

        return s

