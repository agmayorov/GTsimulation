import numpy as np

from gtsimulation.electric_field import GeneralFieldE


class Uniform(GeneralFieldE):

    def __init__(self, E):
        super().__init__()
        self.model_name = "Uniform"
        self.E = np.array(E).astype(np.float64)

    def calc_field(self, *args):
        return self.E

    def __str__(self):
        return (
            f"{self.model_name}\n"
            f"\tE: {self.E} V/m"
        )
