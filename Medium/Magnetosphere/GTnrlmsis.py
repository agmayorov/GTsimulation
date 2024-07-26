import numpy as np
import datetime
import pyproj
from pymsis import msis

from GT import Regions
from Medium import GTGeneralMedium


class GTnrlmsis(GTGeneralMedium):

    def __init__(self, date: datetime.datetime):
        super().__init__()
        self.region = Regions.Magnetosphere
        self.model = "NRLMSIS"
        # self.model_elements = ['He', 'O', 'N2', 'O2', 'Ar', 'H', 'N']
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
        model_output = msis.run(self.date, lon, lat, alt, self.f107, self.f107a, [self.ap])[0]
        self.density = model_output[0] # kg/m3
        # self.element_abundance = model_output[[0, 1, 2, 3, 4, 6, 7]] / np.sum(model_output[[0, 1, 2, 3, 4, 6, 7]])

    def convert_xyz_to_lla(self, x, y, z):
        return self.transformer.transform(x, y, z, radians=False)

    def get_density(self):
        return self.density # g/сm3

    def get_element_abundance(self):
        pass
        # return self.element_abundance

    def __str__(self):
        return f'NRLMSIS'
