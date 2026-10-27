import numpy as np

from gtsimulation.magnetic_field import AbsBfield


class Uniform(AbsBfield):

    def __init__(self, B, **kwargs):
        super().__init__(**kwargs)
        self.ModelName = "Uniform"
        self.B = np.array(B).astype(np.float64)

    def CalcBfield(self, *args, **kwargs):
        return self.B

    def UpdateState(self, new_date):
        pass

    def __str__(self):
        return (
            f"{self.ModelName}\n"
            f"\tB: {self.B} nT"
        )
