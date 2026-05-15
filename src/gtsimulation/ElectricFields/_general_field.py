from abc import ABC, abstractmethod

from gtsimulation.Global import Regions


class GeneralFieldE(ABC):

    def __init__(self):
        self.region = Regions.Undefined
        self.model_name = None

    @abstractmethod
    def calc_field(self, x, y, z):
        pass

    def GetEfield(self, x, y, z):
        Ex, Ey, Ez = self.calc_field(x, y, z)
        return Ex, Ey, Ez

    @abstractmethod
    def to_string(self):
        pass

    def __str__(self):
        return self.to_string()
