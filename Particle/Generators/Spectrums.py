import numpy as np

from Particle.Flux import Flux


class Monolines(Flux):
    def __init__(self, T=1, *args, **kwargs):
        super().__init__(*args, T=T, **kwargs)

    def GenerateEnergySpectrum(self, T):
        self.KinEnergy = np.ones(self.Nevents)*T
