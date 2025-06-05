import numpy as np
from datetime import datetime
from pyproj import Transformer
from pymsis import msis

from gtsimulation.Global import Regions
from gtsimulation.Medium import GTGeneralMedium


class GTnrlmsis(GTGeneralMedium):

    def __init__(self, date: datetime, version=0):
        super().__init__()
        self.region = Regions.Magnetosphere
        self.model = "NRLMSIS"
        self.version = version
        self.output = np.zeros(10)
        self.component_list = ['N2', 'O2', 'O', 'He', 'H', 'Ar', 'N', 'O_anomalous', 'NO']
        self.element_list = ['H', 'He', 'N', 'O', 'Ar']
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
            self.output = np.nan_to_num(msis.run(self.date, lon, lat, alt, self.f107, self.f107a, [self.ap], version=self.version)[0])
        else:
            self.output = np.zeros(10)

    def convert_xyz_to_lla(self, x, y, z):
        return self.transformer.transform(x, y, z, radians=False)

    def get_density(self):
        return self.output[0] # kg/m3

    def get_component_abundance(self):
        c = self.output[1:-1]
        if np.sum(c) > 0:
            c /= np.sum(c)
        return c

    def get_element_list(self):
        return self.element_list

    def get_element_abundance(self):
        c = self.get_component_abundance()
        e = np.array([c[4],
                      c[3],
                      c[0] * 2 + c[6] + c[8],
                      c[1] * 2 + c[2] + c[7] + c[8],
                      c[5]])
        if np.sum(e) > 0:
            e /= np.sum(e)
        return e

    def to_string(self):
        return f"""{self.model}
        Version: {self.version}"""
