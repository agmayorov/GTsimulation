import numpy as np
import nrlmsise00

from Medium import GTGeneralMedium


class GTnrmlsise00(GTGeneralMedium):

    RE = 6371.137e3 # Earth radius in [m]

    def __init__(self, x, y, z, date_time, f107a, f107, ap):
        super().__init__()
        self.region = "Atmosphere"
        self.model = "NRLMSISE-00"
        self.model_elements = ['He', 'O', 'N2', 'O2', 'Ar', 'H', 'N']
        self.f107a = f107a
        self.f107 = f107
        self.ap = ap
        self.calculate_model(x, y, z, date_time)

    def calculate_model(self, x, y, z, date_time, **kwargs):
        alt = (np.sqrt(x ** 2 + y ** 2 + z ** 2) - self.RE) * 1e-3 # altitude in [km]
        lat = 60 # converting (x, y, z) coordinates to latitude in degrees north
        lon = -30 # converting (x, y, z) coordinates to longitude in degrees east
        model_output = nrlmsise00.msise_flat(date_time, alt, lat, lon, self.f107a, self.f107, self.ap)
        self.x = x
        self.y = y
        self.z = z
        self.date_time = date_time
        self.density = model_output[5]
        self.element_abundance = model_output[[0, 1, 2, 3, 4, 6, 7]] / np.sum(model_output[[0, 1, 2, 3, 4, 6, 7]])

    def get_density(self):
        return self.density

    def get_element_abundance(self):
        return self.element_abundance
