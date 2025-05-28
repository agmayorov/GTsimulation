import numpy as np

from MagneticFields import AbsBfield


class Uniform(AbsBfield):

    def __init__(self, B, **kwargs):
        super().__init__(**kwargs)
        self.ModelName = "Uniform"
        self.B = np.array(B)

    def CalcBfield(self, *args, **kwargs):
        return self.B

    def UpdateState(self, new_date):
        pass

    def to_string(self):
        s = f"""{self.ModelName}
            B: {self.B} nT"""
        return s

    def __str__(self):
        return self.to_string()
