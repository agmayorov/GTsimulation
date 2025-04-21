import numpy as np

from Global import Regions
from MagneticFields import AbsBfield


class Uniform(AbsBfield):
    ToMeters = 1

    def __init__(self, B0, **kwargs):
        super().__init__(**kwargs)
        self.ModelName = "Uniform"
        self.B0 = B0
        self.B = np.array([0, 0, self.B0])

    def CalcBfield(self, *args, **kwargs):
        return self.B

    def UpdateState(self, new_date):
        pass


    def to_string(self):
        s = f"""{self.ModelName}
            B0: {self.B0}
            """
        return s

    def __str__(self):
        return self.to_string()