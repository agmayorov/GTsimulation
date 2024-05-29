import numpy as np

from Particle.Flux import Flux
from Particle.functions import ConvertUnits
from Particle.GetNucleiProp import GetNucleiProp


class Monolines(Flux):
    def __init__(self, T=1, *args, **kwargs):
        self.T = T
        super().__init__(*args, **kwargs)

    def GenerateEnergySpectrum(self):
        if isinstance(self.T, (int, float)):
            self.KinEnergy = np.ones(self.Nevents) * self.T
        elif isinstance(self.T, (list, np.ndarray)) and len(self.T) == self.Nevents:
            self.KinEnergy = self.T

    def __str__(self):
        s = f"""Monolines
        Energy: {self.T}"""
        s1 = super().__str__()

        return s + s1


class PowerSpectrum(Flux):
    def __init__(self, EnergyMin=1, EnergyMax=10, RangeUnits='T', Base='T', SpectrumIndex=1., *args, **kwargs):
        self.EnergyMin = EnergyMin
        self.EnergyMax = EnergyMax
        self.SpectrumIndex = SpectrumIndex
        self.RangeUnits = RangeUnits
        self.Base = Base
        super().__init__(*args, **kwargs)

    def GenerateEnergySpectrum(self):
        self.KinEnergy = np.zeros(self.Nevents)
        for s in range(self.Nevents):
            A, Z, M, *_ = GetNucleiProp(self.ParticleNames[s])
            M = M / 1e3  # MeV/c2 -> GeVA, /c2

            EnergyRange = np.array([self.EnergyMin, self.EnergyMax])
            if self.RangeUnits != self.Base:
                EnergyRangeS = ConvertUnits(EnergyRange, self.RangeUnits, self.Base, M, A, Z)
            else:
                EnergyRangeS = EnergyRange
            ksi = np.random.rand()
            if self.SpectrumIndex == -1:
                self.KinEnergy[s] = EnergyRangeS[0] * np.power((EnergyRangeS[1] / EnergyRangeS[0]), ksi)
            else:
                g = self.SpectrumIndex + 1.
                self.KinEnergy[s] = np.power(np.power(EnergyRangeS[0], g) +
                                             ksi * (np.power(EnergyRangeS[1], g) - np.power(EnergyRangeS[0], g)),
                                             (1 / g))

            if self.RangeUnits != self.Base:
                self.KinEnergy[s] = ConvertUnits(self.KinEnergy[s], self.Base, self.RangeUnits, M, A, Z)

    def __str__(self):
        s = f"""PowerSpectrum
        Minimal Energy: {self.EnergyMin}
        Maximal Energy: {self.EnergyMax}
        Spectrum Index: {self.SpectrumIndex}"""
        s1 = super().__str__()

        return s + s1
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


class Uniform(Flux):
    def __init__(self, MinT=1, MaxT=10, *args, **kwargs):
        self.MinT = MinT
        self.MaxT = MaxT
        super().__init__(*args, **kwargs)

    def GenerateEnergySpectrum(self):
        self.KinEnergy = np.random.rand(self.Nevents) * (self.MaxT - self.MinT) + self.MinT

    def __str__(self):
        s = f"""Uniform
        Minimal Energy: {self.MinT}
        Maximal Energy: {self.MaxT}"""
        s1 = super().__str__()

        return s + s1
