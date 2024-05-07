import numpy as np

from Particle.Flux import Flux
from Particle.functions import ConvertUnits
from Particle.GetNucleiProp import GetNucleiProp


class Monolines(Flux):
    def __init__(self, T=1, *args, **kwargs):
        super().__init__(*args, T=T, **kwargs)

    def GenerateEnergySpectrum(self, T):
        self.KinEnergy = np.ones(self.Nevents)*T


class PowerSpectrum(Flux):
    def __init__(self, EnergyMin=1, EnergyMax=10, RangeUnits='T', Base='T', SpectrumIndex=1., *args, **kwargs):
        super().__init__(*args, EnergyMin=EnergyMin, EnergyMax=EnergyMax, RangeUnits=RangeUnits, Base=Base,
                         SpectrumIndex=SpectrumIndex, **kwargs)

    def GenerateEnergySpectrum(self, EnergyMin, EnergyMax, RangeUnits, Base, SpectrumIndex):
        self.KinEnergy = np.zeros(self.Nevents)
        for s in range(self.Nevents):
            A, Z, M, *_ = GetNucleiProp(self.ParticleNames[s])
            M = M / 1e3  # MeV/c2 -> GeVA, /c2

            EnergyRange = np.array([EnergyMin, EnergyMax])
            if RangeUnits != Base:
                EnergyRangeS = ConvertUnits(EnergyRange, RangeUnits, Base, M, A, Z)
            else:
                EnergyRangeS = EnergyRange
            ksi = np.random.rand()
            if SpectrumIndex == -1:
                self.KinEnergy[s] = EnergyRangeS[0] * np.power((EnergyRangeS[1] / EnergyRangeS[0]), ksi)
            else:
                g = SpectrumIndex + 1
                self.KinEnergy[s] = np.power(np.power(EnergyRangeS[0], g) +
                                          ksi * (np.power(EnergyRangeS[1], g) -np.power(EnergyRangeS[0], g)),
                                          (1 / g))

            if RangeUnits != Base:
                self.KinEnergy[s] = ConvertUnits(self.KinEnergy[s], Base, RangeUnits, M, A, Z)
#
# class ForceField(PowerSpectrum):
#     def __init__(self, T=1, *args, **kwargs):
#         super().__init__(*args, T=T, **kwargs)
#
#     def GenerateEnergySpectrum(self, T):
#         self.KinEnergy = np.zeros(self.Nevents)
#         for s in range(self.Nevents):
#             A, Z, M, *_ = GetNucleiProp(self.ParticleNames[s])
#             M = M / 1e3  # MeV/c2 -> GeVA, /c2
