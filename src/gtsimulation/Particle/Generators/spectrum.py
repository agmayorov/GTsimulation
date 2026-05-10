from abc import ABC, abstractmethod

import numpy as np

from gtsimulation.Particle.GeneratorCR import GetGCRflux
from gtsimulation.Particle.functions import convert_units


class AbsSpectrum(ABC):
    def __init__(self, flux_object=None):
        self.flux = flux_object

    @abstractmethod
    def generate_energy_spectrum(self, *args, **kwargs):
        return []

    @abstractmethod
    def to_string(self):
        pass

    def __str__(self):
        return self.to_string()


class ContinuumSpectrum(AbsSpectrum):
    def __init__(self, energy_min=500., energy_max=10000., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.energy_range = np.array([self.energy_min, self.energy_max])

    def generate_energy_spectrum(self, *args, **kwargs):
        return []

    def to_string(self):
        s = f"""
        Minimal Energy: {self.energy_min}
        Maximal Energy: {self.energy_max}"""
        return s


class Monolines(AbsSpectrum):
    def __init__(self, energy=1000., *args, **kwargs):
        self.T = energy
        super().__init__(*args, **kwargs)

    def generate_energy_spectrum(self):
        match self.T:
            case int() | float():
                energy = np.ones(self.flux.Nevents) * self.T
            case list() | np.ndarray():
                energy = np.concatenate((np.tile(self.T, self.flux.Nevents // len(self.T)), self.T[:self.flux.Nevents % len(self.T)]))
            case _:
                raise TypeError('Unsupported type')
        return energy

    def to_string(self):
        s = f"""Monolines
        Energy: {self.T}"""
        return s


class PowerSpectrum(ContinuumSpectrum):
    def __init__(self, energy_min=500., energy_max=10000., energy_range_units='T', base='T', spectrum_index=-1., *args, **kwargs):
        super().__init__(energy_min, energy_max, *args, **kwargs)
        self.energy_range_units = energy_range_units
        self.base = base
        self.spectrum_index = spectrum_index

    def generate_energy_spectrum(self):
        energy = np.zeros(self.flux.Nevents)
        for s in range(self.flux.Nevents):
            z, a, m = self.flux.particles[s].Z, self.flux.particles[s].A, self.flux.particles[s].M
            if self.energy_range_units != self.base:
                energy_range_s = convert_units(self.energy_range, self.energy_range_units, self.base, m, a, z)
            else:
                energy_range_s = self.energy_range

            ksi = np.random.rand()
            if self.spectrum_index == -1:
                energy[s] = energy_range_s[0] * (energy_range_s[1] / energy_range_s[0]) ** ksi
            else:
                g = self.spectrum_index + 1.
                energy[s] = (energy_range_s[0] ** g + ksi * (energy_range_s[1] ** g - energy_range_s[0] ** g)) ** (1 / g)

            if self.energy_range_units != self.base:
                energy[s] = convert_units(energy[s], self.base, self.energy_range_units, m, a, z)
        return energy

    def to_string(self):
        s = f"""PowerSpectrum
        Spectrum Index: {self.spectrum_index}"""
        s_super = super().to_string()
        return s + s_super


class ForceField(ContinuumSpectrum):
    def __init__(self, energy_min=500., energy_max=10000., energy_range_units='T', modulation_potential=500., *args, **kwargs):
        super().__init__(energy_min, energy_max, *args, **kwargs)
        self.energy_range_units = energy_range_units
        self.modulation_potential = modulation_potential

    def generate_energy_spectrum(self):
        energy = np.zeros(self.flux.Nevents)
        # partitioning by particle species
        unique_name, index, index_inverse, count = np.unique(self.flux.name, return_index=True, return_inverse=True, return_counts=True)
        for i, particle in enumerate(unique_name):
            z, a, m = self.flux.particles[index[i]].Z, self.flux.particles[index[i]].A, self.flux.particles[index[i]].M
            if self.energy_range_units != 'T':
                energy_range_s = convert_units(self.energy_range, self.energy_range_units, 'T', m, a, z)
            else:
                energy_range_s = self.energy_range

            f_max = GetGCRflux('T', np.geomspace(energy_range_s[0], energy_range_s[1], 1000) / 1e3, self.modulation_potential / 1e3, particle).max()
            index_particle = np.flatnonzero(index_inverse == i)
            bunch_size = count[i] * 10 # multiplying by 10 for faster calculations
            while True:
                energy_played = np.random.uniform(*energy_range_s, bunch_size)
                ksi = f_max * np.random.rand(bunch_size)
                index_suited = np.flatnonzero(ksi < GetGCRflux('T', energy_played / 1e3, self.modulation_potential / 1e3, particle))
                if index_suited.size < index_particle.size:
                    energy[index_particle[:index_suited.size]] = energy_played[index_suited]
                    index_particle = np.delete(index_particle, range(index_suited.size))
                else:
                    energy[index_particle] = energy_played[index_suited[:index_particle.size]]
                    break

            if self.energy_range_units != 'T':
                energy[index_inverse == i] = convert_units(energy[index_inverse == i], 'T', self.energy_range_units, m, a, z)
        return energy

    def to_string(self):
        s = f"""ForceField
        Modulation Potential: {self.modulation_potential} MV"""
        s_super = super().to_string()
        return s + s_super


class Uniform(ContinuumSpectrum):
    def __init__(self, energy_min=500., energy_max=10000., *args, **kwargs):
        super().__init__(energy_min, energy_max, *args, **kwargs)

    def generate_energy_spectrum(self):
        return np.random.uniform(self.energy_min, self.energy_max, self.flux.Nevents)

    def to_string(self):
        s = f"""Uniform"""
        s_super = super().to_string()
        return s + s_super


class UserInput(AbsSpectrum):
    def __init__(self, energy=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = np.array(energy, ndmin=1)

    def generate_energy_spectrum(self, *args, **kwargs):
        if self.energy.size != self.flux.Nevents:
            raise ValueError("The number of initial energies does not correspond to the number of particles")
        return self.energy

    def to_string(self):
        s = f"""User Input
        Energy size: {self.energy.size}"""
        return s
