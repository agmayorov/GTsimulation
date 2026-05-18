import os
import datetime
import numpy as np
from numba import jit

from gtsimulation.Global import Units, Regions
from gtsimulation.MagneticFields import AbsBfield
from gtsimulation.MagneticFields.Magnetosphere.Functions import transformations


class Dipole(AbsBfield):
    Re = 6371.137e3
    ToMeters = Units.RE2m

    def __init__(self, date: int | datetime.datetime = 0, units="SI_nT", M=None, psi=0, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Magnetosphere
        self.ModelName = "Dipole"
        self.Units = "RE"
        self.Date = date
        self.units = units
        self.psi = psi

        path = os.path.dirname(os.path.realpath(__file__))

        coefs = np.load(f"{path}{os.sep}Data{os.sep}HarmonicCoeffsIGRF.npy", allow_pickle=True).item()
        self.g10sm = np.poly1d(coefs["g10_fit"])
        self.g11sm = np.poly1d(coefs["g11_fit"])
        self.h11sm = np.poly1d(coefs["h11_fit"])

        if M is not None:
            self.M = M
        elif self.Date == 0:
            self.M = 30100
        elif isinstance(self.Date, datetime.datetime):
            self.__SetEarthDipMagMom()

    def UpdateState(self, new_date: datetime.datetime):
        self.Date = new_date
        self.__SetEarthDipMagMom()

    def __SetEarthDipMagMom(self):
        assert self.units in ["SI_nT", "SI", "CGS_G", "CGS", "SEC"]
        assert 1900 <= self.Date.year <= 2021
        ND, N = Dipole.GetNDaysInMonth(self.Date.year, self.Date.month)
        D = self.Date.year + (self.Date.day + np.sum(ND[:self.Date.month - 1]) - 0.5) / np.sum(ND)
        match self.units:
            case "SI_nT":
                Mx = self.g11sm(D)
                My = self.h11sm(D)
                Mz = self.g10sm(D)
            case "SI":
                Mx = (self.Re ** 3 / 1e-7) * self.g11sm(D) / 1e9
                My = (self.Re ** 3 / 1e-7) * self.h11sm(D) / 1e9
                Mz = (self.Re ** 3 / 1e-7) * self.g10sm(D) / 1e9
            case 'CGS_G':
                Mx = self.g11sm(D) / 1e9 * 1e4
                My = self.h11sm(D) / 1e9 * 1e4
                Mz = self.g10sm(D) / 1e9 * 1e4
            case "CGS":
                Mx = (self.Re * 1e2) ** 3 * self.g11sm(D) / 1e9 * 1e4
                My = (self.Re * 1e2) ** 3 * self.h11sm(D) / 1e9 * 1e4
                Mz = (self.Re * 1e2) ** 3 * self.g10sm(D) / 1e9 * 1e4
            case _:
                Mx = (self.Re * 1e2) * self.g11sm(D) / 1e9 * 1e4 * 300 / 1e9
                My = (self.Re * 1e2) * self.h11sm(D) / 1e9 * 1e4 * 300 / 1e9
                Mz = (self.Re * 1e2) * self.g10sm(D) / 1e9 * 1e4 * 300 / 1e9
        self.M = np.sqrt(Mx ** 2 + My ** 2 + Mz ** 2)

    def CalcBfield(self, x, y, z, **kwargs):
        psi = self.psi
        M = self.M
        X, Y, Z = transformations.geo2dipmag(x, y, z, psi, 1)
        return transformations.geo2dipmag(*self.__calcBfield(X, Y, Z, M, psi), psi, 0)

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def __calcBfield(x, y, z, M, psi):
        Q = M / (np.sqrt(x ** 2 + y ** 2 + z ** 2)) ** 5

        Bx = Q * ((y ** 2 + z ** 2 - 2 * x ** 2) * np.sin(psi) - 3 * (z * x) * np.cos(psi))
        By = -3 * y * (Q * (x * np.sin(psi) + z * np.cos(psi)))
        Bz = Q * ((x ** 2 + y ** 2 - 2 * z ** 2) * np.cos(psi) - 3 * (z * x) * np.sin(psi))

        return Bx, By, Bz

    @staticmethod
    def GetNDaysInMonth(year, month):
        ND = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        if year % 4 == 0:
            ND[1] = 29

        return ND, ND[month - 1]

    def to_string(self):
        s = f"""Dipole
        psi: {self.psi}
        MagMom: {self.M}"""
        return s
