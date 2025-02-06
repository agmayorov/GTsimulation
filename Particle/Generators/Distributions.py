from enum import Enum

import numpy as np

from abc import ABC, abstractmethod

class GeneratorModes(Enum):
    Inward = 1
    Outward = -1


class AbsDistribution(ABC):
    def __init__(self, FluxObj=None, *args, **kwargs):
        self.flux = FluxObj

    @abstractmethod
    def GenerateCoordinates(self, *args, **kwargs):
        return [], []


class SphereSurf(AbsDistribution):
    def __init__(self, Radius=0, Center=np.zeros(3), *args, **kwargs):
        self.Center = Center
        self.Radius = Radius
        super().__init__(*args, **kwargs)

    def GenerateCoordinates(self):
        Ro = self.Radius * self.flux.ToMeters
        Rc = self.Center * self.flux.ToMeters
        r_ret, v = [], []
        match self.flux.Mode:
            case GeneratorModes.Inward:
                theta = np.arccos(1 - 2 * np.random.rand(self.flux.Nevents, 1))
                phi = 2 * np.pi * np.random.rand(self.flux.Nevents, 1)
                r = np.concatenate((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1)

                newZ = r
                newX = np.cross(newZ, np.tile([[0, 0, 1]], (self.flux.Nevents, 1)))
                newX /= np.linalg.norm(newX, axis=1)[:, np.newaxis]
                newY = np.cross(newZ, newX)

                S = np.stack((newX.T, newY.T, newZ.T), axis=1)

                ksi = np.random.rand(1, 1, self.flux.Nevents)
                sin_theta = np.sqrt(ksi)
                cos_theta = np.sqrt(1 - ksi)
                phi = 2 * np.pi * np.random.rand(1, 1, self.flux.Nevents)
                p = np.concatenate((-sin_theta * np.cos(phi), -sin_theta * np.sin(phi), -cos_theta))
                r_ret = r * Ro + Rc

                if self.flux.V0 is None:
                    v = np.concatenate([S[:, :, i] @ p[:, :, i] for i in range(self.flux.Nevents)], axis=1).T
                else:
                    v = np.tile(self.flux.V0, (self.flux.Nevents, 1)) / np.linalg.norm(self.flux.V0)

            case GeneratorModes.Outward:
                theta = np.arccos(1 - 2 * np.random.rand(self.flux.Nevents, 1))
                phi = 2 * np.pi * np.random.rand(self.flux.Nevents, 1)
                r = np.concatenate((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1)
                r_ret = r * Ro + Rc

                if self.flux.V0 is None:
                    theta = np.arccos(1 - 2 * np.random.rand(self.flux.Nevents, 1))
                    phi = 2 * np.pi * np.random.rand(self.flux.Nevents, 1)
                    v = np.hstack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                else:
                    v = np.tile(self.flux.V0, (self.flux.Nevents, 1)) / np.linalg.norm(self.flux.V0)

        return r_ret, v

    def __str__(self):
        s = f"""Sphere Surface
        Center: {self.Center}
        Radius: {self.Radius}"""

        return s


class SphereVol(AbsDistribution):
    def __init__(self, Radius=0, Center=np.zeros(3), *args, **kwargs):
        self.Center = Center
        self.Radius = Radius
        super().__init__(*args, **kwargs)

    def GenerateCoordinates(self):
        Ro = self.Radius * self.flux.ToMeters
        Rc = self.Center * self.flux.ToMeters
        theta = np.arccos(1 - 2 * np.random.rand(self.flux.Nevents, 1))
        phi = 2 * np.pi * np.random.rand(self.flux.Nevents, 1)

        max_ro3 = 1/3 * Ro**3
        ro3 = np.random.rand(self.flux.Nevents, 1) * max_ro3
        ro = np.cbrt(3*ro3)

        r = np.concatenate((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1)
        r_ret = r * ro + Rc

        if self.flux.V0 is None:
            theta = np.arccos(1 - 2 * np.random.rand(self.flux.Nevents, 1))
            phi = 2 * np.pi * np.random.rand(self.flux.Nevents, 1)
            v = np.hstack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        else:
            v = np.tile(self.flux.V0, (self.flux.Nevents, 1)) / np.linalg.norm(self.flux.V0)

        return r_ret, v

    def __str__(self):
        s = f"""Sphere Volume
        Center: {self.Center}
        Radius: {self.Radius}"""

        return s


class Disk(AbsDistribution):
    def __init__(self, Radius=15, Width=0.3, *args, **kwargs):
        self.Radius = Radius
        self.Width = Width
        super().__init__(*args, **kwargs)

    def GenerateCoordinates(self):
        Ro = self.Radius * self.flux.ToMeters
        Width = self.Width * self.flux.ToMeters
        z = np.random.rand(self.flux.Nevents, 1)*Width/2 - Width/2
        b_max = 1/2 * Ro**2
        phi = np.random.rand(self.flux.Nevents, 1)*2*np.pi
        b = np.random.rand(self.flux.Nevents, 1) * b_max
        R = np.sqrt(2*b)
        x = R*np.cos(phi)
        y = R*np.sin(phi)
        r = np.hstack((x, y, z))
        if self.flux.V0 is None:
            theta = np.arccos(1 - 2 * np.random.rand(self.flux.Nevents, 1))
            phi = 2 * np.pi * np.random.rand(self.flux.Nevents, 1)
            v = np.hstack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        else:
            v = np.tile(self.flux.V0, (self.flux.Nevents, 1)) / np.linalg.norm(self.flux.V0)

        return r, v

    def __str__(self):
        s = f"""Disk Surface
        Width: {self.Width}
        Radius: {self.Radius}"""

        return s


class UserInput(AbsDistribution):
    def __init__(self, R0=np.zeros((1, 3)), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = np.array(R0, ndmin=2)
        self.v = np.array(self.flux.V0, ndmin=2)

    def GenerateCoordinates(self, *args, **kwargs):
        if self.r.shape != (self.flux.Nevents, 3) or self.v.shape != (self.flux.Nevents, 3):
            raise ValueError("The number of initial coordinates and velocities does not correspond to the number of particles")

        return self.r, self.v

    def __str__(self):
        s = f"""User Input
        R0 shape: {self.r.shape}
        V0 shape: {self.v.shape}"""

        return s

