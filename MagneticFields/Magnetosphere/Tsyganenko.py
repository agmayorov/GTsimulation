import datetime
import numpy as np
import pyspedas

from MagneticFields import AbsBfield


class Tsyganenko(AbsBfield):
    def __init__(self, date=0, ModCode=96):
        super().__init__()
        self.ModelName = "Tsyg"
        self.Region = "M"
        self.Date = date
        self.ModCode = ModCode

        T96 = np.load("MagneticFields/Magnetosphere/T96input_short.npy", allow_pickle=True).item()
        self.T96input = T96["T96input"]
        self.T96_date = T96["date"]

        self.Year, self.DoY, self.Secs, self.DTnum = [0]*4

        if self.Date == 0:
            self.ps = 0
            self.iopt = [1.1200, 2.0000, 1.6000, -0.2000]
        elif isinstance(self.Date, datetime.datetime):
            self.Year = self.Date.year
            self.Doy = self.Date.timetuple().tm_yday
            self.Secs = self.Date.second
            self.DTnum = self.Date.toordinal() + 366

            self.ps = self.GetPsi()
            self.iopt = self.GetTsyganenkoInd()

    def GetPsi(self):
        pass

    def GetTsyganenkoInd(self):
        pass

    def GetBfield(self, x, y, z, **kwargs):
        pass

    def UpdateState(self, new_date):
        pass
