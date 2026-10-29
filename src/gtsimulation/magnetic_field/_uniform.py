import numpy as np

from gtsimulation.magnetic_field import AbsBfield


class Uniform(AbsBfield):

    def __init__(self, B):
        super().__init__()
        self.ModelName = "Uniform"
        self.B = np.array(B).astype(np.float64)

    def calc_field(self, *args):
        return self.B

    def update_state(self, new_date):
        pass

    def __str__(self):
        return (
            f"{self.ModelName}\n"
            f"\tB: {self.B} nT"
        )
