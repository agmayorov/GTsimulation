from enum import Enum
from abc import ABC, abstractmethod

import pyproj


class _AbsRegion(ABC):

    @staticmethod
    def transform(x, y, z, name, units):
        return x, y, z

    @staticmethod
    @abstractmethod
    def additions(*args, **kwargs):
        pass


class _Heliosphere(_AbsRegion):

    @staticmethod
    def additions(*args, **kwargs):
        pass


class _Galaxy(_AbsRegion):

    @staticmethod
    def additions(*args, **kwargs):
        pass


class _Magnetosphere(_AbsRegion):

    @staticmethod
    def additions(*args, **kwargs):
        # TODO Andrey
        pass

    @staticmethod
    def transform(x, y, z, name, units):
        if name == 'LLA':
            # x = lat, y = long, z = altitude
            # units = Units.RE2m or Units.km2m
            # TODO make more rigorous after units addition in the GT
            transformer = pyproj.Transformer.from_crs({"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
                                                      {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})
            # Matlab lla2ecef([lat, long, altitude]) -> python transformer.transform(long, lat, altitude, radians=False)
            x, y, z = transformer.transform(y, x, z*1000, radians=False)
            x, y, z = x/units, y/units, z/units
        return x, y, z


class Regions(Enum):
    Magnetosphere = _Magnetosphere
    Heliosphere = _Heliosphere
    Galaxy = _Galaxy
