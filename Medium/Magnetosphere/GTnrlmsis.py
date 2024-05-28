import numpy as np
import datetime
import pyproj
from pymsis import msis

from Medium import GTGeneralMedium


class GTnrmlsis(GTGeneralMedium):

    def __init__(self, date: datetime.datetime, f107a=150, f107=150, ap=4):
        super().__init__()
        self.region = "Magnetosphere"
        self.model = "NRLMSIS"
        self.model_elements = ['He', 'O', 'N2', 'O2', 'Ar', 'H', 'N']
        self.date = date
        self.f107a = f107a
        self.f107 = f107
        self.ap = ap
        self.transformer = pyproj.Transformer.from_crs({"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
                                                       {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'})

    def calculate_model(self, x, y, z, **kwargs):
        lon, lat, alt = self.convert_xyz_to_lla(x, y, z)
        alt *= 1e-3 # m -> km
        model_output = nrlmsise00.msise_flat(self.date, alt, lat, lon, self.f107a, self.f107, self.ap)
        self.density = model_output[5] # g/сm3
        self.element_abundance = model_output[[0, 1, 2, 3, 4, 6, 7]] / np.sum(model_output[[0, 1, 2, 3, 4, 6, 7]])

    def convert_xyz_to_lla(self, x, y, z):
        return self.transformer.transform(x, y, z, radians=False)

    def get_density(self):
        return self.density # g/сm3

    def get_element_abundance(self):
        return self.element_abundance
