from enum import Enum
from abc import ABC, abstractmethod

from Global import Units, Regions


class AbsBfield(ABC):
    ToMeters = 1

    def __init__(self, use_tesla=False, use_meters=False, **kwargs):
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
