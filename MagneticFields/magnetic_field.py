from enum import Enum
from abc import ABC, abstractmethod


class Regions(Enum):
    Magnetosphere = 1
    Heliosphere = 2


class AbsBfield(ABC):

    def __init__(self):
        self.Region = None
        self.ModelName = None

    @abstractmethod
    def GetBfield(self, x, y, z, **kwargs):
        pass

    @abstractmethod
    def UpdateState(self, new_date):
        pass
