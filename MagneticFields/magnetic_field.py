from enum import Enum
from abc import ABC, abstractmethod


class Units:
    km2m = 1e3

    AU2m = 149.597870700e9
    AU2km = 149.597870700e6

    pc2m = 3.08567758149e16
    kpc2m = 3.08567758149e19

    fm2cm = 1e-13

    RE2m = 6378137.1
    RE2km = 6378.1371
    RM2m = 1737400
    RM2km = 1737.4

    T2nT = 1e9


class Regions(Enum):
    Magnetosphere = 1
    Heliosphere = 2


class AbsBfield(ABC):
    ToMeters = 1

    def __init__(self, use_tesla=False, use_meters=False):
        self.Region = None
        self.ModelName = None
        self.Units = None
        self.use_tesla = use_tesla
        self.use_meters = use_meters

    @abstractmethod
    def CalcBfield(self, x, y, z, **kwargs):
        pass

    @abstractmethod
    def UpdateState(self, new_date):
        pass

    @classmethod
    def FromMeters(cls, x, y, z):
        return x / cls.ToMeters, y / cls.ToMeters, z / cls.ToMeters

    @staticmethod
    def ToTesla(Bx, By, Bz):
        return Bx / Units.T2nT, By / Units.T2nT, Bz / Units.T2nT

    def GetBfield(self, x, y, z, **kwargs):
        if self.use_meters:
            x, y, z = self.FromMeters(x, y, z)
        Bx, By, Bz = self.CalcBfield(x, y, z, **kwargs)
        if self.use_tesla:
            return self.ToTesla(Bx, By, Bz)

        return Bx, By, Bz
