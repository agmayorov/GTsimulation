from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

import pyproj
from numba import jit


class _AbsRegion(ABC):
    SaveAdd = dict()
    calc_additional = False

    @staticmethod
    def transform(x, y, z, name, units):
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



class _Heliosphere(_AbsRegion):
    @staticmethod
    def additions(*args, **kwargs):
        pass

    @classmethod
    def CalcAdditional(cls):
        return cls.calc_additional

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def AdditionalEnergyLosses(r, v, T, M, dt, frwd_tracing, c, ToMeters):
        r = r/ToMeters
        R = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        theta = np.arccos(r[2] / R)
        div_wind = 2/R * (300 + 475 * (1 - np.sin(theta) ** 8))/149.597870700e6
        dE = dt * T/3 * div_wind * (T+2*M)/(T+M)
        T -= frwd_tracing*dE

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
    def transform(x, y, z, name, units):
        if name == 'LLA':
            # x = lat, y = long, z = altitude
            # units = Units.RE2m or Units.km2m
            # TODO make more rigorous after units addition in the GT
            transformer = pyproj.Transformer.from_crs({"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
                                                      {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})
            # Matlab lla2ecef([lat, long, altitude]) -> python transformer.transform(long, lat, altitude, radians=False)
            x, y, z = transformer.transform(y, x, z*1000, radians=False)
            x, y, z = x/units, y/units, z/units
        return x, y, z

    @staticmethod
    def checkSave(Simulator, Nsave):
        Nsave_check = (Simulator.TrackParamsIsOn * Simulator.IsFirstRun * Simulator.TrackParams["GuidingCentre"] * (Nsave != 1))
        assert Nsave_check != 1, "To calculate all additions correctly 'Nsave' parameter must be equal to 1"


class Regions(Enum):
    Magnetosphere = _Magnetosphere
    Heliosphere = _Heliosphere
    Galaxy = _Galaxy
