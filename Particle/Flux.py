from collections.abc import Sequence, Iterable

from Particle.Generators import GeneratorModes
from Particle.Generators import Monolines, SphereSurf
from Particle.Particle import CRParticle


class Flux(Sequence):
    def __init__(self, Spectrum=Monolines, Distribution=SphereSurf, Names='pr', Nevents: int = 1, ToMeters=1, V0=None,
                 Mode: GeneratorModes | str = GeneratorModes.Inward, *args, **kwargs):
        self.Mode = Mode if isinstance(Mode, GeneratorModes) else GeneratorModes[Mode]
        self.Nevents = Nevents
        self.ToMeters = ToMeters
        self.Names = Names
        self.V0 = V0
        self.particles = []
        self.kinetic_energy = None
        self._spectrum = Spectrum(flux_object=self, *args, **kwargs)
        self._distribution = Distribution(FluxObj=self, *args, **kwargs)
        # self.Generate()

    def Generate(self):
        self.GenerateCoordinates()
        self.GenerateParticles(self.Names)
        self.generate_energy_spectrum()
        self.particles = []
        for i in range(self.Nevents):
            self.particles.append(CRParticle(r=self.r[i], v=self.v[i], T=self.kinetic_energy[i], Name=self.ParticleNames[i]))

    def generate_energy_spectrum(self, *args, **kwargs):
        self.kinetic_energy = self._spectrum.generate_energy_spectrum()

    def GenerateParticles(self, Names):
        if isinstance(Names, (Iterable, Sequence)) and not isinstance(Names, str):
            if len(Names) == self.Nevents:
                self.ParticleNames = Names
            else:
                raise Exception("Wrong number of particles")
        else:
            self.ParticleNames = [Names] * self.Nevents

    def GenerateCoordinates(self, *args, **kwargs):
        self.r, self.v = self._distribution.GenerateCoordinates(*args, **kwargs)

    def __getitem__(self, item):
        return self.particles[item]

    def __len__(self):
        return len(self.particles)

    def __str__(self):
        s = f"""
        Number of particles: {self.Nevents}"""
        if self.Names is not None:
            s += f"""
        Particles: {self.Names}"""
        s += f"""
        V: {self.V0 if self.V0 is not None else 'Isotropic'}
        Spectrum: {str(self._spectrum)}
        Distribution: {str(self._distribution)}"""

        return s
