import numpy as np
import datetime
import pyproj
import iri2016

from Medium import GTGeneralMedium


class GTiri2016(GTGeneralMedium):

    def __init__(self, date: datetime.datetime):
        super().__init__()
        self.region = "Magnetosphere"
        self.model = "IRI-2016"
        self.model_elements = ['nO+', 'nN+', 'nH+', 'nHe+', 'nO2+', 'nNO+']
        self.date = date
        self.transformer = pyproj.Transformer.from_crs({"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
                                                       {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'})

    def calculate_model(self, x, y, z):
        lon, lat, alt = self.convert_xyz_to_lla(x, y, z)
        alt *= 1e-3 # m -> km
        model_output = iri2016.IRI(self.date, [alt, alt, 1], lat, lon)
        self.density = [2.656, 2.325, 0.167, 0.664, 5.333, 4.983] @ \
                       model_output[['nO+', 'nN+', 'nH+', 'nHe+', 'nO2+', 'nNO+']].to_array().to_numpy() * 1e-23

    def convert_xyz_to_lla(self, x, y, z):
        return self.transformer.transform(x, y, z, radians=False)

    def get_density(self):
        return self.density[0]

    def get_element_abundance(self):
        # return self.element_abundance
        pass
