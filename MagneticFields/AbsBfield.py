from abc import ABC, abstractmethod


class AbsBfield(ABC):

    def __init__(self):
        self.Region = None
        self.Model = None

    @abstractmethod
    def GetBfield(self, x, y, z, **kwargs):
        pass
