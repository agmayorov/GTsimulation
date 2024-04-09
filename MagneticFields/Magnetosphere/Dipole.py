import datetime
import numpy as np
from MagneticFields import AbsBfield


class Dipole(AbsBfield):
    Re = 6371.137e3

    def __init__(self, date=0, units="SI_nT", M=None, psi=0):
        super().__init__()
        self.Region = "Magnetosphere"
        self.Model = "Dipole"
        self.Date = date
        self.units = units
        self.psi = psi

        coefs = np.load("MagneticFields/Magnetosphere/Data/HarmonicCoeffsIGRF.npy", allow_pickle=True).item()
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
        if self.units == "SI_nT":
            Mx = self.g11sm(D)
            My = self.h11sm(D)
            Mz = self.g10sm(D)
        elif self.units == "SI":
            Mx = (self.Re ** 3 / 1e-7) * self.g11sm(D) / 1e9
            My = (self.Re ** 3 / 1e-7) * self.h11sm(D) / 1e9
            Mz = (self.Re ** 3 / 1e-7) * self.g10sm(D) / 1e9
        elif self.units == 'CGS_G':
            Mx = self.g11sm(D) / 1e9 * 1e4
            My = self.h11sm(D) / 1e9 * 1e4
            Mz = self.g10sm(D) / 1e9 * 1e4
        elif self.units == "CGS":
            Mx = (self.Re * 1e2) ** 3 * self.g11sm(D) / 1e9 * 1e4
            My = (self.Re * 1e2) ** 3 * self.h11sm(D) / 1e9 * 1e4
            Mz = (self.Re * 1e2) ** 3 * self.g10sm(D) / 1e9 * 1e4
        else:
            Mx = (self.Re * 1e2) * self.g11sm(D) / 1e9 * 1e4 * 300 / 1e9
            My = (self.Re * 1e2) * self.h11sm(D) / 1e9 * 1e4 * 300 / 1e9
            Mz = (self.Re * 1e2) * self.g10sm(D) / 1e9 * 1e4 * 300 / 1e9
        self.M = np.sqrt(Mx ** 2 + My ** 2 + Mz ** 2)

    def GetBfield(self, x, y, z, **kwargs):
        Q = self.M / (np.sqrt(x ** 2 + y ** 2 + z ** 2)) ** 5

        Bx = Q * ((y ** 2 + z ** 2 - 2 * x ** 2) * np.sin(self.psi) - 3 * (z * x) * np.cos(self.psi))
        By = -3 * y * (Q * (x * np.sin(self.psi) + z * np.cos(self.psi)))
        Bz = Q * ((x ** 2 + y ** 2 - 2 * z ** 2) * np.cos(self.psi) - 3 * (z * x) * np.sin(self.psi))

        return Bx, By, Bz

    @staticmethod
    def GetNDaysInMonth(year, month):
        ND = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        if year % 4 == 0:
            ND[1] = 29

        return ND, ND[month - 1]


