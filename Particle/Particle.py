import numpy as np

from Particle.NucleiProp import NucleiProp


class Particle:

    def __init__(self, Name=None, Z=None, M=None, PDG=None):
        # Self properties
        if Name is not None:
            if Name in NucleiProp.keys():
                self.A = NucleiProp[Name]['A']
                self.Z = NucleiProp[Name]['Z']
                self.M = NucleiProp[Name]['M']
                self.PDG = NucleiProp[Name]['PDG']
                self.Name = Name
            elif Z is not None and M is not None:
                self.A = None
                self.Z = Z
                self.M = M
                self.PDG = None
                self.Name = Name
            else:
                raise Exception("No such particle")
        elif PDG is not None:
            found = False
            # TODO: make this part more optimal (without loop)
            for key in NucleiProp.keys():
                if NucleiProp[key]['PDG'] == PDG:
                    self.A = NucleiProp[key]['A']
                    self.Z = NucleiProp[key]['Z']
                    self.M = NucleiProp[key]['M']
                    self.PDG = NucleiProp[key]['PDG']
                    self.Name = key
                    found = True
                    break
            if not found:
                raise Exception("No such particle")
        else:
            raise Exception("No such particle")


class CRParticle(Particle):
    def __init__(self, r=np.array([1, 0, 0]), v=np.array([1, 0, 0]), T=1, E=None, **kwargs):
        super().__init__(**kwargs)
        self.coordinates = r
        self.velocities = v / np.linalg.norm(v)

        if T is not None and T > 0:
            self.T = T
            self.E = T + self.M
        elif E is not None and E > 0:
            self.E = E
            self.T = E - self.M
        else:
            raise Exception("Particle without or with negative energy")

    def UpdateState(self, newV, newT, dt):
        self.T = newT
        self.E = self.T + self.M

        self.coordinates += newV*dt
        self.velocities = newV/np.linalg.norm(newV)
