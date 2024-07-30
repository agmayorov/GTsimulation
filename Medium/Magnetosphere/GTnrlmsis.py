import numpy as np
import datetime
import pyproj
from pymsis import msis

from Global import Regions
from Medium import GTGeneralMedium


class GTnrlmsis(GTGeneralMedium):

    def __init__(self, date: datetime.datetime, version=2.1):
        super().__init__()
        self.region = Regions.Magnetosphere
        self.model = "NRLMSIS"
        self.version = version
        self.element_list = ['N2', 'O2', 'O', 'He', 'H', 'Ar', 'N', 'O_anomalous', 'NO']
        self.date = date
        f107, f107a, ap = msis.get_f107_ap(self.date)
        self.f107 = f107
        self.f107a = f107a
        self.ap = ap
        self.transformer = pyproj.Transformer.from_crs({"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
                                                       {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'})

    def calculate_model(self, x, y, z, **kwargs):
        lon, lat, alt = self.convert_xyz_to_lla(x, y, z)
        alt *= 1e-3 # m -> km
        self.model_output = msis.run(self.date, lon, lat, alt, self.f107, self.f107a, [self.ap], version=self.version)[0]

    def convert_xyz_to_lla(self, x, y, z):
        return self.transformer.transform(x, y, z, radians=False)

    def get_density(self):
        return self.model_output[0] # kg/m3

    def get_element_abundance(self):
        return self.model_output[1:-1] / np.nansum(model_output[1:-1])

    def __str__(self):
        return f'NRLMSIS-{self.version}'
