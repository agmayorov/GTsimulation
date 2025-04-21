import numpy as np
import warnings

from enum import Enum
from abc import ABC, abstractmethod
from numba import jit
from pyproj import Transformer

from Particle import CRParticle, Flux
from Particle.Generators import Distributions, Spectrums
from Interaction import G4Shower
warnings.simplefilter("always")


class _AbsRegion(ABC):
    SaveAdd = dict()
    calc_additional = False

    @staticmethod
    def transform(x, y, z, name):
        return x, y, z

    @classmethod
    def set_params(cls, CalcAdditionalEnergy=False):
        cls.calc_additional = CalcAdditionalEnergy

    @staticmethod
    @abstractmethod
    def additions(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def checkSave(*args, **kwargs):
        pass

    @classmethod
    def CalcAdditional(cls):
        return False

    @staticmethod
    def AdditionalEnergyLosses(r, v, T, M, dt, frwd_tracing, c, ToMeters):
        return v, T

    @classmethod
    def ret_str(cls):
        return "\t\tAdditional Energy Losses: False"

    @staticmethod
    def do_before_loop(*args, **kwargs):
        pass

class _Undefined(_AbsRegion):

    @staticmethod
    def additions(*args, **kwargs):
        pass

    @staticmethod
    def checkSave(*args, **kwargs):
        pass


class _Heliosphere(_AbsRegion):
    @staticmethod
    def additions(*args, **kwargs):
        pass

    @classmethod
    def CalcAdditional(cls):
        return cls.calc_additional

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def AdditionalEnergyLosses(r, v, T, M, dt, frwd_tracing, c):
        r = r / 149.597870700e9  # meters to au
        R = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        theta = np.arccos(r[2] / R)
        div_wind = 2 / R * (300 + 475 * (1 - np.sin(theta) ** 8)) / 149.597870700e6  # km/s to au/s
        dE = dt * T / 3 * div_wind * (T + 2 * M) / (T + M)
        T -= frwd_tracing * dE

        V = c * np.sqrt((T + M) ** 2 - M ** 2) / (T + M)
        Vn = np.linalg.norm(v)
        v *= V / Vn
        return v, T

    @classmethod
    def ret_str(cls):
        return f"\t\tAdditional Energy Losses: {cls.calc_additional}"


class _Galaxy(_AbsRegion):

    @staticmethod
    def additions(*args, **kwargs):
        pass


class _Magnetosphere(_AbsRegion):
    SaveAdd = {"Invariants": False, "PitchAngles": False, "MirrorPoints": False, "Lshell": False,
               "GuidingCentre": False}

    @staticmethod
    def additions(*args, **kwargs):
        # TODO Andrey
        pass

    @staticmethod
    def transform(x, y, z, name):
        if name == 'LLA':
            # x = lat, y = long, z = altitude
            # units = Units.RE2m or Units.km2m
            # TODO make more rigorous after units addition in the GT
            transformer = Transformer.from_crs({"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
                                               {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})
            # Matlab lla2ecef([lat, long, altitude]) -> python transformer.transform(long, lat, altitude, radians=False)
            x, y, z = transformer.transform(y, x, z, radians=False)
            # x, y, z = x/units, y/units, z/units
        return x, y, z

    @staticmethod
    def checkSave(Simulator, Nsave):
        Nsave_check = (Simulator.TrackParamsIsOn * Simulator.IsFirstRun * Simulator.TrackParams["GuidingCentre"] * (
                    Nsave != 1))
        assert Nsave_check != 1, "To calculate all additions correctly 'Nsave' parameter must be equal to 1"

    @staticmethod
    def do_before_loop(simulator, gen, prod_tracks):
        if simulator.InteractNUC is not None and gen > 1:
            particle = simulator.Particles[simulator.index]
            r = np.array(particle.coordinates)
            V_normalized = np.array(particle.velocities)
            T = particle.T
            geo_to_lla = Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                                              {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'})
            lon, lat, alt = geo_to_lla.transform(r[0], r[1], r[2], radians=False)
            angle = np.arccos(np.dot(-V_normalized, r / np.linalg.norm(r))) / np.pi * 180
            if 0 < alt < 80e3 and angle < 70:
                primary, secondary = G4Shower(particle.PDG, T, r, V_normalized, simulator.Date)
                simulator.IsPrimDeath = True
                if secondary.size > 0 and gen < simulator.InteractNUC['GenMax']:
                    if simulator.Verbose:
                        print(f"EAS ~ {secondary.size} secondaries")
                        print(secondary)
                    for p in secondary:
                        PDGcode_p = p["PDGcode"]
                        # Try to find a particle (TODO: REMOVE IN THE FUTURE)
                        try:
                            name_p = CRParticle(PDG=PDGcode_p, Name=None).Name
                        except:
                            warnings.warn(f"Particle with code {PDGcode_p} was not found. Calculation is skipped.")
                            continue
                        params = simulator.ParamDict.copy()
                        params["Particles"] = Flux(
                            Distribution=Distributions.UserInput(R0=p['Position'], V0=p['MomentumDirection']),
                            Spectrum=Spectrums.UserInput(energy=p['KineticEnergy']),
                            Names=name_p
                        )
                        if PDGcode_p in [12, 14, 16, 18, -12, -14, -16, -18]:
                            params["Medium"] = None
                            params["InteractNUC"] = None
                        new_process = simulator.__class__(**params)
                        new_process._GTSimulator__gen = gen + 1
                        track = new_process.CallOneFile()[0]
                        track["Particle"]["R0"] = p['VertexPosition']
                        track["Particle"]["V0"] = p['VertexMomentumDirection']
                        track["Particle"]["T0"] = p['VertexKineticEnergy']
                        prod_tracks.append(track)


class Regions(Enum):
    Magnetosphere = _Magnetosphere
    Heliosphere = _Heliosphere
    Galaxy = _Galaxy
    Undefined = _Undefined
