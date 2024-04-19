import datetime
import numpy as np

from MagneticFields import AbsBfield, Regions
from MagneticFields.Magnetosphere.Functions import transformations, t89, t96
from MagneticFields.magnetic_field import Units


class Tsyganenko(AbsBfield):
    def __init__(self, date=0, ModCode=96, **kwargs):
        super().__init__(**kwargs)
        self.ModelName = "Tsyg"
        self.Region = Regions.Magnetosphere
        self.Date = date
        self.ModCode = ModCode

        T96 = np.load("MagneticFields/Magnetosphere/Data/T96input_short.npy", allow_pickle=True).item()
        self.T96input = T96["T96input"]
        self.T96_date = T96["date"]

        self.Year, self.DoY, self.Secs, self.DTnum = 1, 1, 0, 1

        if self.Date == 0:
            self.ps = 0
            self.iopt = [1.1200, 2.0000, 1.6000, -0.2000]
            self.Date = datetime.datetime(1, 1, 1)
        elif isinstance(self.Date, datetime.datetime):
            self.Year = self.Date.year
            self.Doy = self.Date.timetuple().tm_yday
            self.Secs = self.Date.second
            self.DTnum = self.Date.toordinal()  # + 366

            self.ps = self.GetPsi()
            self.iopt = self.GetTsyganenkoInd()

    def GetPsi(self):
        [x, y, z] = transformations.geo2mag_eccentric(0, 0, 1, 0, datetime.datetime.fromordinal(self.DTnum))
        [x, y, z] = transformations.gei2geo(x, y, z, self.Year, self.Doy, self.Secs, 0)
        [x, y, z] = transformations.gei2gsm(x, y, z, self.Year, self.Doy, self.Secs, 1)
        psi = np.arccos(z / np.linalg.norm([x, y, z]))

        return psi

    def GetTsyganenkoInd(self):

        ia = np.where((self.T96_date[:, 0] == self.Date.year) * (self.T96_date[:, 1] == self.Date.month) *
                      (self.T96_date[:, 3] == self.Date.day) * (self.T96_date[:, 4] == self.Date.hour))[0][0]

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
        pass

    @staticmethod
    def FromMeters(x, y, z):
        return x/Units.RE2m, y/Units.RE2m, z/Units.RE2m
