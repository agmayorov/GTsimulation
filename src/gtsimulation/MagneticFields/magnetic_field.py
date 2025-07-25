from abc import ABC, abstractmethod

from gtsimulation.Global import Units, Regions


class AbsBfield(ABC):
    ToMeters = 1

    def __init__(self, use_tesla=True, use_meters=True, **kwargs):
        self.Region = Regions.Undefined
        self.ModelName = None
        self.Units = 'm'
        self.use_tesla = use_tesla
        self.use_meters = use_meters

    @abstractmethod
    def CalcBfield(self, x, y, z, **kwargs):
        pass

    @abstractmethod
    def UpdateState(self, new_date):
        pass

    @classmethod
    def from_meters(cls, x, y, z):
        return x / cls.ToMeters, y / cls.ToMeters, z / cls.ToMeters

    @staticmethod
    def to_tesla(Bx, By, Bz):
        return Bx / Units.T2nT, By / Units.T2nT, Bz / Units.T2nT

    def GetBfield(self, x, y, z, **kwargs):
        if self.use_meters:
            x, y, z = self.from_meters(x, y, z)
        Bx, By, Bz = self.CalcBfield(x, y, z, **kwargs)
        if self.use_tesla:
            return self.to_tesla(Bx, By, Bz)
        return Bx, By, Bz

    @abstractmethod
    def to_string(self):
        pass

    def __str__(self):
        return self.to_string()
