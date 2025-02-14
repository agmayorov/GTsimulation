import datetime
import numpy as np

from Global import Units, Regions
from MagneticFields import AbsBfield
from MagneticFields.Magnetosphere.Functions import transformations, t89, t96, gauss


class Tsyganenko(AbsBfield):
    ToMeters = Units.RE2m

    def __init__(self, date: int | datetime.datetime = 0, ModCode=96, **kwargs):
        super().__init__(**kwargs)
        self.ModelName = "Tsyganenko"
        self.Region = Regions.Magnetosphere
        self.Units = "RE"
        self.Date = date
        self.ModCode = ModCode

        T96 = np.load("MagneticFields/Magnetosphere/Data/T96input_short.npy", allow_pickle=True).item()
        self.T96input = T96["T96input"]
        self.T96_date = T96["date"]

        self.Year, self.DoY, self.Secs, self.DTnum = 1, 1, 0, 1

        self.__SetPsiInd()

    def __SetPsiInd(self):
        if self.Date == 0:
            self.ps = 0
            self.iopt = [1.1200, 2.0000, 1.6000, -0.2000]
            self.Date = datetime.datetime(1, 1, 1)
        elif isinstance(self.Date, datetime.datetime):
            self.Year = self.Date.year
            self.DoY = self.Date.timetuple().tm_yday
            self.Secs = self.Date.second + 60 * self.Date.minute + 3600 * self.Date.hour
            self.DTnum = self.Date.toordinal()  # + 366

            self.ps = self.GetPsi()
            self.iopt = self.GetTsyganenkoInd()

    def GetPsi(self):
        self.g, self.h, _ = gauss.LoadGaussCoeffs("MagneticFields/Magnetosphere/IGRF13/igrf13coeffs.npy", self.Date)
        [x, y, z] = transformations.geo2mag_eccentric(0, 0, 1, 0, self.g, self.h)
        [x, y, z] = transformations.gei2geo(x, y, z, self.Year, self.DoY, self.Secs, 0)
        [x, y, z] = transformations.gei2gsm(x, y, z, self.Year, self.DoY, self.Secs, 1)
        psi = np.arccos(z / np.linalg.norm([x, y, z]))

        return psi[0]

    def GetTsyganenkoInd(self):

        ia = np.where((self.T96_date[:, 0] == self.Date.year) * (self.T96_date[:, 1] == self.Date.month) *
                      (self.T96_date[:, 2] == self.Date.day) * (self.T96_date[:, 3] == self.Date.hour))[0][0]

        ind = None
        if self.ModCode == 89:
            ind = self.T96input[ia, -1]
        elif self.ModCode == 96:
            ind = self.T96input[ia, :4]

        return ind

    def CalcBfield(self, x, y, z, **kwargs):
        X, Y, Z = transformations.geo2gsm(x, y, z, self.Year, self.DoY, self.Secs, 1)
        Bx, By, Bz = 0, 0, 0
        if self.ModCode == 89:
            Bx, By, Bz = t89.t89(self.iopt, self.ps, X, Y, Z)
        elif self.ModCode == 96:
            Bx, By, Bz = t96.t96(self.iopt, self.ps, X, Y, Z)

        Bx, By, Bz = transformations.geo2gsm(Bx, By, Bz, self.Year, self.DoY, self.Secs, 0)

        return Bx, By, Bz

    def UpdateState(self, new_date):
        self.Date = new_date
        self.__SetPsiInd()

    def to_string(self):
        s = f"""Tsyganenko
        Mode: {self.ModCode}
        pis: {self.ps}"""

        return s

