import datetime
import numpy as np
from MagneticFields.AbsBfield import AbsBfield


class Dipole(AbsBfield):
    Re = 6371.137e3

    def __init__(self, date=0, units="SI_nT", M=None, psi=0):
        super().__init__()
        self.Region = "Magnetosphere"
        self.Model = "Dipole"
        self.psi = psi
        if M is not None:
            self.M = M
        elif date == 0:
            self.M = 30100
        elif isinstance(date, datetime.date):
            self.SetEarthDipMagMom(date, units)

    def SetEarthDipMagMom(self, date, units):
        assert units in ["SI_nT", "SI", "CGS_G", "CGS", "SEC"]
        assert 1900 <= date.year <= 2021
        coefs = np.load("MagneticFields/Magnetosphere/HarmonicCoeffsIGRF.npy", allow_pickle=True).item()
        g10sm = np.poly1d(coefs["g10_fit"])
        g11sm = np.poly1d(coefs["g11_fit"])
        h11sm = np.poly1d(coefs["h11_fit"])
        ND, N = Dipole.GetNDaysInMonth(date.year, date.month)
        D = date.year + (date.day + np.sum(ND[:date.month - 1]) - 0.5) / np.sum(ND)
        if units == "SI_nT":
            Mx = g11sm(D)
            My = h11sm(D)
            Mz = g10sm(D)
        elif units == "SI":
            Mx = (self.Re ** 3 / 1e-7) * g11sm(D) / 1e9
            My = (self.Re ** 3 / 1e-7) * h11sm(D) / 1e9
            Mz = (self.Re ** 3 / 1e-7) * g10sm(D) / 1e9
        elif units == 'CGS_G':
            Mx = g11sm(D) / 1e9 * 1e4
            My = h11sm(D) / 1e9 * 1e4
            Mz = g10sm(D) / 1e9 * 1e4
        elif units == "CGS":
            Mx = (self.Re * 1e2) ** 3 * g11sm(D) / 1e9 * 1e4
            My = (self.Re * 1e2) ** 3 * h11sm(D) / 1e9 * 1e4
            Mz = (self.Re * 1e2) ** 3 * g10sm(D) / 1e9 * 1e4
        else:
            Mx = (self.Re * 1e2) * g11sm(D) / 1e9 * 1e4 * 300 / 1e9
            My = (self.Re * 1e2) * h11sm(D) / 1e9 * 1e4 * 300 / 1e9
            Mz = (self.Re * 1e2) * g10sm(D) / 1e9 * 1e4 * 300 / 1e9
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


