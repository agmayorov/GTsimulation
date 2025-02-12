import numpy as np
import datetime
from pyproj import Transformer
from pymsis import msis

from Global import Regions
from Medium import GTGeneralMedium


class GTnrlmsis(GTGeneralMedium):

    def __init__(self, date: datetime.datetime, version=0):
        super().__init__()
        self.region = Regions.Magnetosphere
        self.model = "NRLMSIS"
        self.version = version
        self.model_output = np.zeros(10)
        self.element_list = ['N2', 'O2', 'O', 'He', 'H', 'Ar', 'N', 'O_anomalous', 'NO']
        self.chemical_element_list = ['H', 'He', 'N', 'O', 'Ar']
        self.date = date
        f107, f107a, ap = msis.get_f107_ap(self.date)
        self.f107 = f107
        self.f107a = f107a
        self.ap = ap
        self.transformer = Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                                                {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'})

    def calculate_model(self, x, y, z, **kwargs):
        lon, lat, alt = self.convert_xyz_to_lla(x, y, z)
        alt *= 1e-3 # m -> km
        if alt > 0:
            self.model_output = np.nan_to_num(msis.run(self.date, lon, lat, alt, self.f107, self.f107a, [self.ap], version=self.version)[0])
        else:
            self.model_output = np.zeros(10)

    def convert_xyz_to_lla(self, x, y, z):
        return self.transformer.transform(x, y, z, radians=False)

    def get_density(self):
        return self.model_output[0] # kg/m3

    def get_element_abundance(self):
        e = self.model_output[1:-1]
        if np.sum(e) > 0:
            e /= np.sum(e)
        return e

    def get_chemical_element_abundance(self):
        e = self.get_element_abundance()
        c = np.array([e[4],
                      e[3],
                      e[0] * 2 + e[6] + e[8],
                      e[1] * 2 + e[2] + e[7] + e[8],
                      e[5]])
        if np.sum(c) > 0:
            c /= np.sum(c)
        return c

    def __str__(self):
        return f"NRLMSIS-{self.version}"
