import copy
from collections.abc import Sequence, Iterable

import numpy as np

from gtsimulation.Particle.Generators import AbsDistribution, AbsSpectrum, GeneratorModes
from gtsimulation.Particle.Particle import CRParticle


class Flux(Sequence):
    def __init__(self, Spectrum: AbsSpectrum, Distribution: AbsDistribution, Names=None, PDGcode=None, Nevents: int = 1,
                 V0=None, Mode: GeneratorModes | str = GeneratorModes.Inward, *args, **kwargs):
        self.Mode = Mode if isinstance(Mode, GeneratorModes) else GeneratorModes[Mode]
        self.Nevents = Nevents
        self.name = Names
        self.pdg_code = PDGcode
        self.V0 = V0
        self.particles = []
        self.r = []
        self.v = []
        self.kinetic_energy = None
        self._spectrum = Spectrum
        self._spectrum.flux = self
        self._distribution = Distribution
        self._distribution.flux = self

    def generate(self):
        self.particles.clear()
        self.generate_particles()
        self.generate_coordinates()
        self.generate_energy_spectrum()
        for i in range(self.Nevents):
            self.particles[i].coordinates = self.r[i]
            self.particles[i].velocities = self.v[i]
            self.particles[i].T = self.kinetic_energy[i]
            self.particles[i].E = self.kinetic_energy[i] + self.particles[i].M

    def generate_particles(self):
        if self.name is None and self.pdg_code is None:
            self.name = ['proton'] * self.Nevents
            self.pdg_code = [2212] * self.Nevents
            proton = CRParticle(PDG=2212)
            self.particles = [copy.deepcopy(proton) for _ in range(self.Nevents)]
        elif self.name is not None and self.pdg_code is None:
            if isinstance(self.name, (Iterable, Sequence)) and not isinstance(self.name, str):
                if len(self.name) == self.Nevents:
                    unique_name, index_inverse = np.unique(self.name, return_inverse=True)
                    unique_particles = [CRParticle(Name=name) for name in unique_name]
                    self.particles = [copy.deepcopy(unique_particles[index]) for index in index_inverse]
                    self.pdg_code = [particle.PDG for particle in self.particles]
                else:
                    raise Exception("Wrong number of particles")
            else:
                cr_particle = CRParticle(Name=self.name)
                self.particles = [copy.deepcopy(cr_particle) for _ in range(self.Nevents)]
                self.pdg_code = [cr_particle.PDG] * self.Nevents
        else:
            if isinstance(self.pdg_code, (Iterable, Sequence)) and not isinstance(self.pdg_code, int):
                if len(self.pdg_code) == self.Nevents:
                    unique_pdg_code, index_inverse = np.unique(self.pdg_code, return_inverse=True)
                    unique_particles = [CRParticle(PDG=pdg_code) for pdg_code in unique_pdg_code]
                    self.particles = [copy.deepcopy(unique_particles[index]) for index in index_inverse]
                    self.name = [particle.Name for particle in self.particles]
                else:
                    raise Exception("Wrong number of particles")
            else:
                cr_particle = CRParticle(PDG=self.pdg_code)
                self.particles = [copy.deepcopy(cr_particle) for _ in range(self.Nevents)]
                self.name = [cr_particle.Name] * self.Nevents

    def generate_coordinates(self, *args, **kwargs):
        self.r, self.v = self._distribution.generate_coordinates(*args, **kwargs)

    def generate_energy_spectrum(self, *args, **kwargs):
        self.kinetic_energy = self._spectrum.generate_energy_spectrum(*args, **kwargs)

    def __getitem__(self, item):
        return self.particles[item]

    def __len__(self):
        return len(self.particles)

    def to_string(self):
        s = f"""
        Number of particles: {self.Nevents}"""
        if self.name is not None:
            s += f"""
        Particles: {np.unique(self.name)}"""
        s += f"""
        V: {self.V0 if self.V0 is not None else 'Isotropic'}
        Spectrum: {str(self._spectrum)}
        Distribution: {str(self._distribution)}"""
        return s

    def __str__(self):
        return self.to_string()


class FluxPitchPhase(Flux):
    def __init__(self, Bfield, Pitch=None, Phase=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Pitch = Pitch
        self.Phase = Phase
        self.Bfield = Bfield

    def to_string(self):
        s = super().to_string()
        s1 = f"""
        Pitch Angle: {self.Pitch} [rad]
        Phase Angles: {self.Phase} [rad]"""
        return s + s1


class GyroCenterFlux(Flux):
    def __init__(self, Bfield, Pitchd, Phased, CooGyr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coo_gyr = CooGyr

        if isinstance(Pitchd, np.ndarray):
            if len(Pitchd) == self.Nevents:
                self.pitchd = Pitchd[:, np.newaxis]
            else:
                raise Exception("Wrong number of pitch angles")
        else:
            self.pitchd = np.array([Pitchd] * self.Nevents)[:, np.newaxis]

        if isinstance(Phased, np.ndarray):
            if len(Phased) == self.Nevents:
                self.phased = Phased[:, np.newaxis]
            else:
                raise Exception("Wrong number of phase angles")
        else:
            self.phased = np.array([Phased] * self.Nevents)[:, np.newaxis]

        bfield = Bfield.copy()
        bfield.use_tesla = False
        bfield.use_meters = True
        self.B = bfield.GetBfield(*self.coo_gyr)
        self.Bm = np.linalg.norm(self.B)

    def generate(self):
        self.generate_particles()
        self.generate_energy_spectrum()
        masses = np.array([particle.M for particle in self.particles])
        charges = np.array([particle.Z for particle in self.particles])

        r_lar = self.larmor(self.kinetic_energy, self.Bm, masses, charges, self.pitchd.flatten())[:, np.newaxis]
        B3 = self.B/self.Bm
        B1 = np.copy(B3)
        B1[0] = -(B1[1] ** 2 + B1[2] ** 2) / B1[0]
        B1 /= np.linalg.norm(B1)
        B2 = np.cross(B3, B1)
        init_v = B1 + np.tan(self.phased * np.pi / 180) * B2 + 1 / np.tan(self.pitchd * np.pi / 180) * B3
        init_v /= np.linalg.norm(init_v)
        offset = ((np.cross(init_v, self.B) / np.linalg.norm(np.cross(init_v, self.B), axis=1)[:, np.newaxis]) * r_lar)
        init_coo = self.coo_gyr - offset
        self.v = init_v
        self.r = init_coo
        for i in range(self.Nevents):
            self.particles[i].coordinates = self.r[i]
            self.particles[i].velocities = self.v[i]
            self.particles[i].T = self.kinetic_energy[i]
            self.particles[i].E = self.kinetic_energy[i] + self.particles[i].M

    @staticmethod
    def larmor(T, Bm, M, Z, pitchd):
        """
        :param T: kinetic energy in MeV
        :param Bm: Magnetic field intensity in nT
        :param M:  mass in MeV
        :param Z: charge number
        :param pitchd: pitch angle in degree

        :return: Output larmor radius in m
        """
        cc = 299_792_458.
        return (np.sqrt((T + M) ** 2 - M ** 2) * 1e6 / cc * np.sin(pitchd / 180 * np.pi)) / (Z * Bm * 1e-9)

    def to_string(self):
        s = super().to_string()
        s1 = f"""
        GyroCenter Coordinates: {self.coo_gyr} [m]
        Pitch Angle: {self.pitchd} [deg]
        Phase Angles: {self.phased} [deg]"""
        return s + s1
