from particle.Particle import CRParticle

class Flux:
    def __init__(self, r, v, Energy, ParticleNames):
        self.__r = r
        self.__v = v
        self.__E = Energy
        self.__ParticleNames = ParticleNames
        self.__flux = []

        for i in range(len(ParticleNames)):
            self.__flux.append(CRParticle(r=r[i], v=v[i], E=Energy[i], Type=ParticleNames[i]))

    def __getitem__(self, item):
        return self.__flux[item]

    @property
    def E(self):
        return self.__E
