import numpy as np

from gtsimulation.ElectricFields import GeneralFieldE


class UniformFieldE(GeneralFieldE):

    def __init__(self, E):
        super().__init__()
        self.model_name = "Uniform"
        self.E = np.array(E).astype(np.float64)

    def calc_field(self, *args):
        return self.E

    def to_string(self):
        s = f"""{self.model_name}
            E: {self.E} V/m"""
        return s
