from enum import Enum
from collections.abc import Sequence, Iterable
from abc import ABC, abstractmethod
from typing import SupportsIndex

import numpy as np

from Particle.Particle import CRParticle


class GeneratorModes(Enum):
    Inward = 1
    Outward = -1


# TODO: add others functions of distribution of energy
class Flux(Sequence, ABC):
    def __init__(self, Names='pr', Mode: GeneratorModes = GeneratorModes.Inward, Radius=1, Center=np.zeros(3),
                 Nevents: int = 1, *args, **kwargs):
        self.Mode = Mode
        self.Nevents = Nevents
        self.GenerateCoordinates(Radius, Center)
        self.GenerateEnergySpectrum(*args, **kwargs)
        self.GenerateParticles(Names)

        self.particles = []

        for i in range(self.Nevents):
            self.particles.append(CRParticle(r=self.r[i], v=self.v[i], T=self.KinEnergy[i], Name=self.ParticleNames[i]))

    @abstractmethod
    def GenerateEnergySpectrum(self, *args, **kwargs):
        self.KinEnergy = []

    def GenerateParticles(self, Names):
        if isinstance(Names, (Iterable, Sequence)) and not isinstance(Names, str):
            if len(Names) == self.Nevents:
                self.ParticleNames = Names
            else:
                raise Exception("Wrong number of particles")
        else:
            self.ParticleNames = [Names] * self.Nevents

    def GenerateCoordinates(self, Ro, Rc):
        match self.Mode:
            case GeneratorModes.Inward:
                theta = np.arccos(1 - 2 * np.random.rand(self.Nevents, 1))
                phi = 2 * np.pi * np.random.rand(self.Nevents, 1)
                r = np.concatenate((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1)

                newZ = r
                newX = np.cross(newZ, np.tile([[0, 0, 1]], (self.Nevents, 1)))
                newX /= np.linalg.norm(newX, axis=1)[:, np.newaxis]
                newY = np.cross(newZ, newX)

                S = np.stack((newX.T, newY.T, newZ.T), axis=1)

                ksi = np.random.rand(1, 1, self.Nevents)
                sin_theta = np.sqrt(ksi)
                cos_theta = np.sqrt(1 - ksi)
                phi = 2 * np.pi * np.random.rand(1, 1, self.Nevents)
                p = np.concatenate((-sin_theta * np.cos(phi), -sin_theta * np.sin(phi), -cos_theta))
                self.r = r * Ro + Rc
                self.v = np.concatenate([S[:, :, i] @ p[:, :, i] for i in range(self.Nevents)], axis=1).T

            case GeneratorModes.Outward:
                self.r = np.tile(Rc, (self.Nevents, 1))
                theta = np.arccos(1 - 2 * np.random.rand(self.Nevents, 1))
                phi = 2 * np.pi * np.random.rand(self.Nevents, 1)
                v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    def __getitem__(self, item):
        return self.particles[item]

    def __len__(self):
        return len(self.particles)
