import numpy as np
from datetime import datetime
from pyproj import Transformer
import pymsis
from pymsis.utils import get_f107_ap

from gtsimulation.common import Regions
from gtsimulation.medium import GTGeneralMedium


class GTnrlmsis(GTGeneralMedium):
    """
    Earth's neutral atmosphere described by the NRLMSIS empirical model.

    The model is evaluated with the ``pymsis`` package at the position given in
    the GEO coordinate system. The F10.7 and Ap indices are taken for the given
    date when the object is created; the date and indices do not change during
    the simulation.

    Parameters
    ----------
    date : datetime.datetime
        Date and time used for the model and for the F10.7 and Ap indices.
    version : {0, 2.0, 2.1}, optional
        NRLMSIS version: 0 for NRLMSISE-00, 2.0 or 2.1 for NRLMSIS 2.x.
        Default is 0.

    Notes
    -----
    calculate_model must be called before the density and abundances are
    requested; below the Earth's surface (geodetic altitude <= 0) the density
    and all abundances are zero.

    get_density returns the total mass density in kg/m3.
    get_element_abundance returns mass fractions of H, He, N, O and Ar
    (see element_list), obtained from the number densities of N2, O2, O, He,
    H, Ar, N, anomalous O and NO. Components not modelled at the given altitude
    are treated as zero.
    """

    def __init__(self, date: datetime, version=0):
        super().__init__()
        self.region = Regions.Magnetosphere
        self.model = "NRLMSIS"
        self.version = version
        self.output = np.zeros(11)  # mass density, 9 number densities, temperature
        self.component_list = ['N2', 'O2', 'O', 'He', 'H', 'Ar', 'N', 'O_anomalous', 'NO']
        self.element_list = ['H', 'He', 'N', 'O', 'Ar']
        self.element_mass = np.array([1.008, 4.0026, 14.007, 15.999, 39.948])  # g/mol
        self.date = date
        f107, f107a, ap = get_f107_ap(self.date)
        self.f107 = f107
        self.f107a = f107a
        self.ap = ap
        self.transformer = Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                                                {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'})

    def calculate_model(self, x, y, z, **kwargs):
        """
        Evaluate the model at the given point.

        The result is stored in the object and is returned by the get_* methods.

        Parameters
        ----------
        x, y, z : float
            Coordinates of the point in GEO, in metres.
        **kwargs
            Not used; accepted for compatibility with the medium interface.
        """
        lon, lat, alt = self.transformer.transform(x, y, z, radians=False)
        alt *= 1e-3  # m -> km
        if alt > 0:
            output = pymsis.calculate(self.date, lon, lat, alt, self.f107, self.f107a, self.ap, version=self.version)
            self.output = np.nan_to_num(output[0])
        else:
            self.output = np.zeros(11)

    def get_density(self):
        """
        Return the total mass density at the last evaluated point.

        Returns
        -------
        float
            Mass density, in kg/m3.
        """
        return self.output[0]

    def get_component_abundance(self):
        """
        Return the fractions of the atmospheric components by number of particles.

        Returns
        -------
        numpy.ndarray, shape (9,)
            Number fractions of the components in the order of component_list.
        """
        c = self.output[1:-1]
        if np.sum(c) > 0:
            c = c / np.sum(c)
        return c

    def get_element_list(self):
        """
        Return the chemical symbols of the elements forming the atmosphere.

        Returns
        -------
        list of str
            Element symbols in the order used by get_element_abundance.
        """
        return self.element_list

    def get_element_abundance(self):
        """
        Return the mass fractions of the elements at the last evaluated point.

        Returns
        -------
        numpy.ndarray, shape (5,)
            Mass fractions of the elements in the order of element_list.
        """
        c = self.get_component_abundance()
        e = np.array([c[4],
                      c[3],
                      c[0] * 2 + c[6] + c[8],
                      c[1] * 2 + c[2] + c[7] + c[8],
                      c[5]]) * self.element_mass
        if np.sum(e) > 0:
            e /= np.sum(e)
        return e

    def __str__(self):
        return (
            f"{self.model}\n"
            f"\tVersion: {self.version}"
        )
