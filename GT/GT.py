import os
import math

import matplotlib.pyplot as plt
import tqdm
from timeit import default_timer as timer
import numpy as np
import importlib
import datetime
import copy
import warnings
warnings.simplefilter("always")

from abc import ABC, abstractmethod

from numba import jit

from Interaction import G4Interaction, G4Decay, SynchCounter, RadLossStep
from MagneticFields.Magnetosphere import Functions, Additions
from Global import Constants, Units, Regions, BreakCode, BreakIndex, SaveCode, SaveDef, BreakDef, \
    BreakMetric, SaveMetric, vecRotMat
from Particle import ConvertT2R, GetAntiParticle, Flux, CRParticle


class GTSimulator(ABC):
    """
    Description

    :param Region: The region of the in which the simulation is taken place. The parameter may have values as
                   `Global.Region.Magnetosphere`, `Global.Region.Heliosphere`, `Global.Region.Galaxy`.
                   See :py:mod:`Global.regions`. If list the second element defines region specific parameters. See
                   :py:mod:`Global.regions._AbsRegion.set_params`. Example: for the heliosphere one may pass
                   `{"CalcAdditionalEnergy": True}` which will take into account the adibatic energy losses of the
                   particles.
    :type Region: Global.Region or list

    :param Bfield: The name of the magnetic field. It should be inside the package :py:mod:`MagneticFields` in the
                   corresponding `Region`. To add some parameters use `list` format and pass the parameters as a `dict`.
                   Example `[NAME, {"param1": value1, "param2": value2}]`
    :type Bfield: str or list

    :param Efield: Electrical field. Similar to :ref:`Bfield`
    :type Efield: str or list

    :param Medium: The medium where particles may go into an interaction. Syntax similar to :ref:`Bfield`.
                   See :py:mod:`Medium`.
    :type Medium: str or list

    :param Date: Date that is used to initialize the fields
    :type Date: datetime.datetime

    :param RadLosses: a `bool` flag that turns the calculations of radiation losses of the particles on.
    :type RadLosses: bool

    :param Particles: The parameter is responsible for the initial particle flux generation. Its value defines the
                      energetic spectrum of the particles. Other parameters the type of particles, their number, e.t.c.
                      The structure is similar  to :ref:`Bfield`. The list of available energy spectrums is available
                      here :py:mod:`Particle.Generators.Spectrums`. For more information regarding flux also
                      see :py:mod:`Particle.Flux`.

                      **Note**: that instead of `Center` parameter `Transform` may be passed
                      its value has the following form `[Name of the coordinate system, [coordinates]]`. Example
                      `["LLA", [60, 70, 1000]]` (lat [degree], long [degree], alt [meters]).
                      See :py:mod:`Global.regions._AbsRegion.transform` for available transforms to a given region.
    :type Particles: str or list

    :param TrackParams: a 'bool' flag that turns the calculations of additional parameters in given region.
                If 'True' all the additional parameters will be calculated. Other parameters that one doesn't need
                to calculate, can be turned off by passing a 'list' instead of 'bool'
                where the second element is a 'dict' that's key is the parameter name and value is 'False'.
                Example: [True, {'GuidingCentre': False}]. See :py:mod:`Global.regions._AbsRegion.SaveAdd`
                for available parameters to a given region.

                **Note**: to calculate some of the additional parameters others must be calculated as well.
    :type TrackParams: bool or list

    :param ParticleOrigin: a 'bool' flag that turns the calculations of particle's origin through the backtracing
    :type ParticleOrigin: bool

    :param IsFirstRun:
    :param ForwardTrck: 1 refers to forward tracing, and -1 to the back tracing
    :type ForwardTrck: 1 or -1

    :param Save: The number of steps that are saved. If the value is 0, then only staring
                 and finishing points are saved. The default parameters that are saved are `Coordinates` and
                 `Velocities`. Other parameters that one needs to save as well, can be turned on by passing a `list`
                 instead of `int` where the second element is a `dict` that's key is the parameter name and value is
                 `True`. Example `[10, {"Clock": True}]`. To available parameters are listed in
                 :py:mod:`Global.codes.SaveCode`.
    :type Save: int or list

    :param Num: The number of simulation steps
    :type Num: int

    :param Step: The time step of simulation in seconds
    :type Step: float

    :param Nfiles: Number of files. If :ref:`Particles` creates a flux of `Nevents` particles then the total number of
                   particles that are going to be simulated is `Nevents`x`Nfiles`.
    :type Nfiles: int

    :param Output: If `None` no files are saved. Otherwise, the name of the saved *.npy* file. If :ref:`Nfiles` is
                   greater than 1. Then the names of the saved files will have the following form `"Output"_i.npy`
    :type Output: str or None

    :param Verbose: If `True` logs are printed
    :type Verbose: bool

    :param BreakCondition: If `None` no break conditions are applied. Otherwise, a `dict` with a key corresponding to
                           the `BreakCondition` name and value corresponding to its value is passed. Example:
                           `{"Rmax": 10}`. In the example the maximal radius of the particle is 10 (in
                           :py:mod:`MagneticFields` distance units). See the full list of break conditions
                           :py:mod:`Global.codes.BreakCode`.
    :type BreakCondition: dict or None

    :param BCcenter: A 3d array of point (in :py:mod:`MagneticFields` distance units). It represents the **(0, 0, 0)**
                     relative to which `Rmax/Rmin, Xmax/Xmin, Ymax/Ymin, Zmax/Zmin` are calculated.
                     Default `np.array([0, 0, 0])`.
    :type BCcenter: np.ndarray

    :param UseDecay: If `True` the particles may decay. Otherwise, not.
    :type UseDecay: bool

    :param InteractNUC:

    :return: dict
    A dictionary is saved. It has the following keys.

    1. Track: The parameters that are saved along the trajectory. See :py:mod:`Global.codes.SaveCode`.
    2. BC: The parameters regarding the simulation end.
        2.1. WOut: The code of break. See :py:mod:`Global.codes.BreakIndex`

        2.2. lon_total:

        2.3. status
    3. Particle: Information about the particle
        3.1. PDG: Its PDG code

        3.2. M: The mass in MeV

        3.3. Z: The charge of the particle in e units.

        3.4. T0: Its initial kinetic energy in MeV

        3.5. Gen: Its generation
    4. Additions: Additional parameters calculated in defined region. See :py:mod:`Global.regions._AbsRegion.SaveAdd`
        4.1. In magnetosphere:
            Invariants:
                I1: First adiabatic invariant along the trajectory of a particle

                I2: Second adiabatic invariant between each pair of reflections at mirror points
            PitchAngles:
                Pitch: Pitch angles along the trajectory of a particle

                PitchEq: Equatorial pitch angles
            MirrorPoints:
                NumMirror: Indexes of trajectory where particle turns to be at a mirror point

                NumEqPitch: Indexes of trajectory where particle crosses the magnetic equator

                NumB0:

                Hmirr: The value of the magnetic field at the mirror points

                Heq: The value of the magnetic field at the magnetic equator
            Lshell:
                L: L-shell calculated on the basis of second invariant and the field at the mirror point

                Lgen: L-shell calculated at every magnetic equator point

            GuidingCentre:
                LR: Larmor radius of a particle

                LRNit:

                Rline: Coordinates of the field line of the guiding centre of the particle

                Bline: The value of the magnetic field of the field line of the guiding centre of the particle

                Req: Coordinates of the guiding centre of the particle calculated from the field line

                Beq: The value of the magnetic field at the magnetic equator calculated from the field line

                BB0: The ratio of the value of the magnetic field at the position of the guiding center corresponding to
                the initial value of the coordinate and at the magnetic equator

                L: L-shell calculated from the field line of the guiding centre

                parReq: Coordinates of the guiding centre of the particle from the local field line

                parBeq: The value of the magnetic field of the local field line of the particle

                parBB0: The ratio of the value of the magnetic field at the position of

                parL: L-shell calculated from the local field line
        4.2. In heliosphere:

        4.3. In galaxy:

    5. Child: List of secondary particles. They have the same parameters.
    """
    def __init__(self, Bfield=None, Efield=None, Region=Regions.Magnetosphere, Medium: None | dict = None,
                 Date=datetime.datetime(2008, 1, 1), RadLosses=False, Particles: None | dict = None, TrackParams=False,
                 ParticleOrigin=False, IsFirstRun=True, ForwardTrck=None, Save: int | list = 1, Num: int = 1e6, Step=1,
                 Nfiles=1, Output=None, Verbose=False, BreakCondition: None | dict = None,
                 BCcenter=np.array([0, 0, 0]), UseDecay=False, InteractNUC: None | dict = None): #TODO: merge BreakCondition and BCcenter

        self.__names = self.__init__.__code__.co_varnames[1:]
        self.__vals = []
        for self.__v in self.__names:
            self.__vals.append(eval(self.__v))
        self.ParamDict = dict(zip(self.__names, self.__vals))

        del self.__names, self.__vals

        self.Verbose = Verbose
        if self.Verbose:
            print("Creating simulator object...")

        self.Date = Date
        if self.Verbose:
            print(f"\tDate: {self.Date}")
            print()

        self.Step = Step
        self.Num = int(Num)

        if self.Verbose:
            print(f"\tTime step: {self.Step}")
            print(f"\tNumber of steps: {self.Num}")
            print()

        self.__SetUseRadLosses(RadLosses)
        if self.Verbose:
            print()

        self.__SetRegion(Region)

        self.ParticleOrigin = ParticleOrigin
        self.ParticleOriginIsOn = False
        self.TrackParamsIsOn = False
        self.TrackParams = self.Region.value.SaveAdd
        if self.ParticleOrigin:
            self.ParticleOriginIsOn = True
            TrackParams = True

        self.__SetAdditions(TrackParams)

        if self.TrackParamsIsOn:
            if not isinstance(Save, list):
                Save = [Save, {"Bfield": True}]
            elif "Bfield" not in Save[1]:
                Save[1] = Save[1] | {"Bfield": True}

        self.IsFirstRun = IsFirstRun

        self.ToMeters = 1
        self.Bfield = None
        self.Efield = None
        self.__SetEMFF(Bfield, Efield)
        if self.Verbose:
            print()

        self.Medium = None
        self.__SetMedium(Medium)
        if self.Verbose:
            print()

        self.UseDecay = False
        self.InteractNUC = None
        self.__gen = 1
        self.__SetNuclearInteractions(UseDecay, InteractNUC)
        if self.Verbose:
            print()

        self.Particles = None
        self.ForwardTracing = 1
        self.__SetFlux(Particles, ForwardTrck)
        if self.Verbose:
            print()

        self.Nfiles = 1 if Nfiles is None or Nfiles == 0 else Nfiles
        self.Output = Output
        self.Npts = 2
        self.Save = SaveDef.copy()
        if self.Verbose:
            print(f"\tNumber of files: {self.Nfiles}")
            print(f"\tOutput file name: {self.Output}_file_num.npy")
        self.__SetSave(Save)
        if self.Verbose:
            print()

        self.__brck_index = BreakCode.copy()
        self.__brck_index.pop("Loop")
        self.__index_brck = BreakIndex.copy()
        self.__brck_arr = BreakDef.copy()
        self.__SetBrck(BreakCondition, BCcenter)

        self.index = 0
        if self.Verbose:
            print("Simulator created!\n")

    def refreshParams(self, Bfield=None, Efield=None, Region=Regions.Magnetosphere, Medium=None,
                 Date=datetime.datetime(2008, 1, 1),
                 RadLosses=False, Particles=dict(), TrackParams=False, ParticleOrigin=False, IsFirstRun=True,
                 ForwardTrck=None, Save: int | list = 1, Num: int = 1e6,
                 Step=1, Nfiles=1, Output=None, Verbose=False, BreakCondition: None | dict = None,
                 BCcenter=np.array([0, 0, 0]), UseDecay=False,
                 InteractNUC: None | dict = None):  # TODO: merge BreakCondition and BCcenter

        if Particles is None:
            Particles = {}
        self.__names = self.__init__.__code__.co_varnames[1:]
        self.__vals = []
        for self.__v in self.__names:
            self.__vals.append(eval(self.__v))
        self.ParamDict = dict(zip(self.__names, self.__vals))

        del self.__names, self.__vals

        self.Verbose = Verbose
        if self.Verbose:
            print("Creating simulator object...")

        self.Date = Date
        if self.Verbose:
            print(f"\tDate: {self.Date}")
            print()

        self.Step = Step
        self.Num = int(Num)

        if self.Verbose:
            print(f"\tTime step: {self.Step}")
            print(f"\tNumber of steps: {self.Num}")
            print()

        self.__SetUseRadLosses(RadLosses)
        if self.Verbose:
            print()

        self.__SetRegion(Region)

        self.ParticleOrigin = ParticleOrigin
        self.ParticleOriginIsOn = False
        self.TrackParamsIsOn = False
        self.TrackParams = self.Region.value.SaveAdd
        if self.ParticleOrigin:
            self.ParticleOriginIsOn = True
            TrackParams = True

        self.__SetAdditions(TrackParams)

        if self.TrackParamsIsOn:
            if not isinstance(Save, list):
                Save = [Save, {"Bfield": True}]
            elif "Bfield" not in Save[1]:
                Save[1] = Save[1] | {"Bfield": True}

        self.IsFirstRun = IsFirstRun

        self.ToMeters = 1
        self.Bfield = None
        self.Efield = None
        self.__SetEMFF(Bfield, Efield)
        if self.Verbose:
            print()

        self.Medium = None
        self.__SetMedium(Medium)
        if self.Verbose:
            print()

        self.UseDecay = False
        self.InteractNUC = None
        self.__gen = 1
        self.__SetNuclearInteractions(UseDecay, InteractNUC)
        if self.Verbose:
            print()

        self.Particles = None
        self.ForwardTracing = 1
        self.__SetFlux(Particles, ForwardTrck)
        if self.Verbose:
            print()

        self.Nfiles = 1 if Nfiles is None or Nfiles == 0 else Nfiles
        self.Output = Output
        self.Npts = 2
        self.Save = SaveDef.copy()
        if self.Verbose:
            print(f"\tNumber of files: {self.Nfiles}")
            print(f"\tOutput file name: {self.Output}_file_num.npy")
        self.__SetSave(Save)
        if self.Verbose:
            print()

        self.__brck_index = BreakCode.copy()
        self.__brck_index.pop("Loop")
        self.__index_brck = BreakIndex.copy()
        self.__brck_arr = BreakDef.copy()
        self.__SetBrck(BreakCondition, BCcenter)

        self.index = 0
        if self.Verbose:
            print("Simulator created!\n")

    def __SetUseRadLosses(self, RadLosses):
        if isinstance(RadLosses, bool):
            self.UseRadLosses = [RadLosses, False]

        if isinstance(RadLosses, list) and (RadLosses[0] == True) and (RadLosses[1]["Photons"] == False):
            self.UseRadLosses = [True, False]

        if isinstance(RadLosses, list) and (RadLosses[0] == True) and (RadLosses[1]["Photons"] == True):
            MinMax = np.array([0, np.inf])
            if "MinE" in RadLosses[1]:
                if RadLosses[1]["MinE"] > 0:
                    MinMax[0] = RadLosses[1]["MinE"]
            if "MaxE" in RadLosses[1]:
                if RadLosses[1]["MaxE"] > 0:
                    MinMax[1] = RadLosses[1]["MaxE"]
            self.UseRadLosses = [True, True, MinMax]

        self.UseRadLosses
        if self.Verbose:
            print(f"\tRadiation Losses: {self.UseRadLosses[0]}")
            print(f"\tSynchrotron Emission: {self.UseRadLosses[1]}")

    def __SetAdditions(self, TrackParams):
        if not isinstance(TrackParams, list):
            self.TrackParamsIsOn = TrackParams
            if self.TrackParamsIsOn:
                self.TrackParams.update((key, True) for key in self.TrackParams)
        else:
            self.TrackParamsIsOn = TrackParams[0]
            if self.TrackParamsIsOn:
                self.TrackParams.update((key, True) for key in self.TrackParams)
            for add in TrackParams[1].keys():
                assert add in self.TrackParams.keys(), f'No such option as "{add}" is allowed'
                self.TrackParams[add] = TrackParams[1][add]

    def __SetNuclearInteractions(self, UseDecay, UseInteractNUC):
        self.UseDecay = UseDecay
        if self.Medium is None and UseInteractNUC is not None:
            raise ValueError('Nuclear Interaction is enabled but Medium is not set')
        self.InteractNUC = UseInteractNUC
        self.IntPathDen = 10 # g/cm2
        if self.Verbose:
            print(f"\tDecay: {self.UseDecay}")
            print(f"\tNuclear Interactions: {self.InteractNUC}")

    def __SetBrck(self, Brck, center):
        if Brck is not None:
            for key in Brck.keys():
                self.__brck_arr[self.__brck_index[key]] = Brck[key]
        if self.Verbose:
            print("\tBreak Conditions: ")
            for key in self.__brck_index.keys():
                print(f"\t\t{key}: {self.__brck_arr[self.__brck_index[key]]}")
            print(f"\tBC center: {center}")
        self.__brck_arr[BreakMetric] *= self.ToMeters
        self.BCcenter = center * self.ToMeters

    def __SetMedium(self, medium):
        if self.Verbose:
            print("\tMedium: ", end='')
        if medium is not None:
            module_name = f"Medium.{self.Region.name}"
            m = importlib.import_module(module_name)
            class_name = medium if not isinstance(medium, list) else medium[0]
            params = {"date": self.Date, **({} if not isinstance(medium, list) else medium[1])}
            if hasattr(m, class_name):
                class_medium = getattr(m, class_name)
                self.Medium = class_medium(**params)
                if self.Verbose:
                    print(str(self.Medium))
            else:
                raise Exception("No such medium")
        else:
            if self.Verbose:
                print(None)

    def __SetFlux(self, flux, forward_trck):
        if self.Verbose:
            print("\tFlux: ", end='')
        module_name = f"Particle.Generators"
        m = importlib.import_module(module_name)
        ToMeters = self.ToMeters
        spectrum = flux.pop("Spectrum", None)
        if spectrum is not None:
            if hasattr(m, spectrum):
                spectrum = getattr(m, spectrum)
                flux["Spectrum"] = spectrum
            else:
                raise Exception("No spectrum")

        distribution = flux.pop("Distribution", None)
        if distribution is not None:
            if hasattr(m, distribution):
                distribution = getattr(m, distribution)
                flux["Distribution"] = distribution
            else:
                raise Exception("No Distribution")
        transform = flux.pop("Transform", None)
        if transform is not None:
            center = flux.get("Center", None)
            assert center is None
            center = self.Region.value.transform(*transform[1], transform[0], ToMeters)
            flux["Center"] = np.array(center)
        params = {"ToMeters": ToMeters, **flux}
        self.Particles = Flux(**params)

        if self.Verbose:
            print(str(self.Particles))
        if forward_trck is not None:
            self.ForwardTracing = forward_trck
            return

        self.ForwardTracing = self.Particles.Mode.value
        if self.Verbose:
            print(f"\tTracing: {'Inward' if self.ForwardTracing == 1 else 'Outward'}")

    def __SetEMFF(self, Bfield=None, Efield=None):
        if self.Verbose:
            print("\tElectric field: ", end='')
        if Efield is not None:
            pass
        else:
            if self.Verbose:
                print(None)

        if self.Verbose:
            print("\tMagnetic field: ", end='')
        if Bfield is not None:
            module_name = f"MagneticFields.{self.Region.name}"
            m = importlib.import_module(module_name)
            class_name = Bfield if not isinstance(Bfield, list) else Bfield[0]
            params = {"date": self.Date, "use_tesla": True, "use_meters": True,
                      **({} if not isinstance(Bfield, list) else Bfield[1])}
            if hasattr(m, class_name):
                B = getattr(m, class_name)
                self.Bfield = B(**params)
                self.ToMeters = self.Bfield.ToMeters
                if self.Verbose:
                    print(str(self.Bfield))
            else:
                raise Exception("No such field")
        else:
            if self.Verbose:
                print(None)

    def __SetRegion(self, Region):
        if not isinstance(Region, list):
            self.Region = Region
        else:
            self.Region = Region[0]
            self.Region.value.set_params(**Region[1])

        if self.Verbose:
            print(f"\tRegion: {self.Region.name}")
            print(self.Region.value.ret_str())
            print()

    def __SetSave(self, Save):
        Nsave = Save if not isinstance(Save, list) else Save[0]

        self.Region.value.checkSave(self, Nsave)

        self.Npts = math.ceil(self.Num / Nsave) if Nsave != 0 else 1
        self.Nsave = Nsave
        if self.Verbose:
            print(f"\tSave every {self.Nsave} step of:")
            print("\t\tCoordinates: True")
            print("\t\tVelocities: True")
        if isinstance(Save, list):
            for saves in Save[1].keys():
                self.Save[saves] = Save[1][saves]
        if self.Verbose:
            for saves in self.Save.keys():
                print(f"\t\t{saves}: {self.Save[saves]}")

    def __call__(self):
        Track = []
        if self.Verbose:
            print("Launching simulation...")
        for i in range(self.Nfiles):
            if self.IsFirstRun:
                print(f"\tFile No {i + 1} of {self.Nfiles}")
            if self.Output is not None:
                file = self.Output.split(os.sep)
                folder = os.sep.join(file[:-1])
                if len(file) != 1 and not os.path.isdir(folder):
                    os.mkdir(folder)
                with open(f'{self.Output}_params.txt', 'w') as file:
                    file.write(str(self.ParamDict))

            RetArr = self.CallOneFile()

            if self.Output is not None:
                if self.Nfiles == 1:
                    np.save(f"{self.Output}.npy", RetArr)
                else:
                    np.save(f"{self.Output}_{i}.npy", RetArr)
                if self.Verbose:
                    print("\tFile saved!")
            else:
                Track.append(RetArr)
        if self.Verbose:
            print("Simulation completed!")
        if self.Output is None:
            return Track

    def CallOneFile(self):
        self.Particles.Generate()
        RetArr = []
        status = "Done"

        SaveE = self.Save["Efield"]
        SaveB = self.Save["Bfield"]
        SaveA = self.Save["Angles"]
        SaveP = self.Save["Path"]
        SaveD = self.Save["Density"]
        SaveC = self.Save["Clock"]
        SaveT = self.Save["Energy"]

        Gen = self.__gen
        GenMax = 1 if self.InteractNUC is None else self.InteractNUC.get("GenMax", 1)

        UseAdditionalEnergyLosses = self.Region.value.CalcAdditional()

        for self.index in range(len(self.Particles)):
            if self.Verbose:
                print("\t\tStarting event...")
            TotTime, TotPathLen, TotPathDen = 0, 0, 0
            if self.Medium is not None and self.InteractNUC is not None:
                LocalDen, LocalChemComp, nLocal, LocalPathDen = 0, np.zeros(len(self.Medium.chemical_element_list)), 0, 0
                LocalPathDenVector = np.empty(0)
                LocalCoordinate = np.empty([0, 3])
            lon_total, lon_prev, full_revolutions = np.array([[0.]]), np.array([[0.]]), 0
            particle = self.Particles[self.index]
            Saves = np.zeros((self.Npts + 1, 17))
            BrckArr = self.__brck_arr
            BCcenter = self.BCcenter
            tau = self.UseDecay * particle.tau
            rnd_dec = 0
            self.IsPrimDeath = False
            prod_tracks = []
            if tau:
                rnd_dec = np.random.rand()
                if self.Verbose:
                    print(f"\t\t\tUse Decay: {self.UseDecay}")
                    print(f"\t\t\tDecay rnd: {rnd_dec}")

            if self.ForwardTracing == -1:
                if self.Verbose:
                    print('\t\t\tBackTracing mode is ON')
                    print('\t\t\tRedefinition of particle on antiparticle')
                GetAntiParticle(particle)
                particle.velocities = -particle.velocities

            Q = particle.Z * Constants.e
            M = particle.M
            E = particle.E
            T = particle.T

            # TODO: optimize calculation of neutral particles
            if Q == 0:              # !!! TEMPORARY FOR FASTER CALCULATIONS !!!
                self.Step *= 1e2    # !!! TEMPORARY FOR FASTER CALCULATIONS !!!

            r = np.array(particle.coordinates)

            V_normalized = np.array(particle.velocities) # unit vector of velosity (beta vector)
            V_norm = Constants.c * np.sqrt(E ** 2 - M ** 2) / E # scalar speed [m/s]
            Vm = V_norm * V_normalized  # vector of velocity [m/s]

            if self.Verbose:
                print(f"\t\t\tParticle: {particle.Name} (M = {M} [MeV], "
                      f"Z = {self.Particles[self.index].Z})")
                print(f"\t\t\tEnergy: {T} [MeV], Rigidity: "
                      f"{ConvertT2R(T, M, particle.A, particle.Z) / 1000 if particle.Z !=0 else np.inf} [GV]")
                print(f"\t\t\tCoordinates: {r / self.ToMeters} [{self.Bfield.Units}]")
                print(f"\t\t\tVelocity: {V_normalized}")
                print(f"\t\t\tbeta: {V_norm / Constants.c}")
                print(f"\t\t\tbeta*dt: {V_norm * self.Step / 1000} [km] / "
                      f"{V_norm * self.Step / self.ToMeters} [{self.Bfield.Units}]")

            # Calculation of EAS for magnetosphere
            self.Region.value.do_before_loop(self, Gen, prod_tracks)

            q = self.Step * Q / 2 / (M * Units.MeV2kg) if M != 0 else 0
            brk = BreakCode["Loop"]
            Step = self.Step
            Num = self.Num
            Nsave = self.Nsave if self.Nsave != 0 else Num + 1
            i_save = 0
            st = timer()

            if self.UseRadLosses[1]:
                synch_record = SynchCounter()
            else:
                synch_record = 0

            if self.Verbose:
                print(f"\t\t\tCalculating: ", end=' ')
            for i in range(Num):
                PathLen = V_norm * Step

                Vp, Yp, Ya, B, E = self.AlgoStep(T, M, q, Vm, r)
                if self.UseRadLosses[1]:
                    synch_record.add_iteration(T,
                                               np.array(self.Bfield.GetBfield(r[0], r[1], r[2])),
                                               Vm, Step)
                Vm, T, new_photons, synch_record = RadLossStep.MakeRadLossStep(Vp, Vm, Yp, Ya, M, Q, r,
                                                                               self, particle, Constants, synch_record)
                prod_tracks.extend(new_photons)

                if UseAdditionalEnergyLosses:
                    Vm, T = self.Region.value.AdditionalEnergyLosses(r, Vm, T, M, Step, self.ForwardTracing,
                                                                     Constants.c, self.ToMeters)

                V_norm, r_new, TotPathLen, TotTime = self.Update(PathLen, Step, TotPathLen, TotTime, Vm, r)

                if self.Medium is not None:
                    self.Medium.calculate_model(r_new[0], r_new[1], r_new[2])
                    Den = self.Medium.get_density() # kg/m3
                    PathDen = (Den * 1e-3) * (PathLen * 1e2) # g/cm2
                    TotPathDen += PathDen # g/cm2
                    if self.InteractNUC is not None and Den > 0:
                        LocalDen += Den
                        LocalChemComp += self.Medium.get_chemical_element_abundance()
                        nLocal += 1
                        LocalPathDen += PathDen
                        LocalPathDenVector = np.append(LocalPathDenVector, LocalPathDen)
                        LocalCoordinate = np.append(LocalCoordinate, r[None, :], axis=0)

                # Decay
                if tau:
                    lifetime = tau * (T/M + 1)
                    if rnd_dec > np.exp(-TotTime/lifetime):
                        self.__Decay(Gen, GenMax, T, TotTime, V_norm, Vm, particle, prod_tracks, r)
                        self.IsPrimDeath = True

                # Nuclear Interaction
                if self.InteractNUC is not None and LocalPathDen > self.IntPathDen and not self.IsPrimDeath:
                    # Construct Rotation Matrix & Save velosity before possible interaction
                    rotationMatrix = vecRotMat(np.array([0, 0, 1]), Vm / V_norm)
                    primary, secondary = G4Interaction(particle.PDG, T, LocalPathDen, (LocalDen * 1e-3) / nLocal, LocalChemComp / nLocal)
                    T = primary['KineticEnergy']
                    V_norm = Constants.c * np.sqrt(1 - (M / (T + M))**2)
                    Vm = V_norm * rotationMatrix @ primary['MomentumDirection']
                    if T > 0 and T > 1: # Cut particles with T < 1 MeV
                        # Only ionization losses
                        LocalDen, LocalChemComp, nLocal, LocalPathDen = 0, np.zeros(len(self.Medium.chemical_element_list)), 0, 0
                        LocalPathDenVector = np.empty(0)
                        LocalCoordinate = np.empty([0, 3])
                    else:
                        # Death due to ionization losses or nuclear interaction
                        self.IsPrimDeath = True
                        if secondary.size > 0 and Gen < GenMax:
                            if self.Verbose:
                                print(f"Nuclear interaction ~ {primary['LastProcess']} ~ {secondary.size} secondaries ~ {np.sum(secondary['KineticEnergy'])} MeV")
                                print(secondary)
                            # Cordinates of interaction point in XYZ
                            path_den_cylinder = (np.linalg.norm(primary['Position']) * 1e2) * (LocalDen * 1e-3 / nLocal) # Path in cylinder [g/cm2]
                            r_interaction = LocalCoordinate[np.argmax(LocalPathDenVector > path_den_cylinder), :]
                            # Parameters for recursive call of GT
                            params = self.ParamDict.copy()
                            params["Date"] += datetime.timedelta(seconds=TotTime)
                            for p in secondary:
                                V_p = rotationMatrix @ p['MomentumDirection']
                                # TODO: possible wrong using of rotation matrix for secondary particles
                                # because this rotation matrix do not correspond to r_interaction point
                                T_p = p['KineticEnergy']
                                PDGcode_p = p["PDGcode"]
                                # Try to find a particle (TODO: REMOVE IN THE FUTURE)
                                try:
                                    name_p = CRParticle(PDG=PDGcode_p, Name=None).Name
                                except:
                                    warnings.warn(f"Particle with code {PDGcode_p} was not found. Calculation is skipped.")
                                    continue
                                params["Particles"] = {"Names": name_p,
                                                       "T": T_p,
                                                       "Center": r_interaction / self.ToMeters,
                                                       "Radius": 0,
                                                       "V0": V_p}
                                new_process = self.__class__(**params)
                                new_process.__gen = Gen + 1
                                prod_tracks.append(new_process.CallOneFile()[0])

                if i % Nsave == 0 or i == Num - 1 or i_save == 0:
                    self.SaveStep(r_new, V_norm, TotPathLen, TotPathDen, TotTime, Vm, i_save, r, T, E, B, Saves,
                                  SaveCode["Coordinates"], SaveCode["Velocities"], SaveCode["Efield"],
                                  SaveCode["Bfield"], SaveCode["Angles"], SaveCode["Path"], SaveCode["Density"],
                                  SaveCode["Clock"], SaveCode["Energy"],
                                  SaveE,
                                  SaveB,
                                  SaveA,
                                  SaveP,
                                  SaveD,
                                  SaveC,
                                  SaveT)
                    i_save += 1
                r = r_new

                # TODO reduce time of the calculation
                # Full revolution
                if self.ParticleOriginIsOn or self.__brck_arr[self.__brck_index["MaxRev"]] != BreakDef[-1]:
                    a_, b_, _ = Functions.transformations.geo2mag_eccentric(r[0], r[1], r[2], 1, self.ParamDict["Date"])
                    lon_total, lon_prev, full_revolutions = Additions.AddLon(lon_total, lon_prev, full_revolutions, i, a_, b_)

                # if i % (self.Num // 100) == 0:
                brck = self.CheckBreak(r, Saves[0, :3], BCcenter, TotPathLen, TotTime, full_revolutions, BrckArr)
                brk = brck[1]
                if brck[0] or self.IsPrimDeath:
                    if brk != -1:
                        self.SaveStep(r_new, V_norm, TotPathLen, TotPathDen, TotTime, Vm, i_save, r, T, E, B, Saves,
                                      SaveCode["Coordinates"], SaveCode["Velocities"], SaveCode["Efield"],
                                      SaveCode["Bfield"], SaveCode["Angles"], SaveCode["Path"], SaveCode["Density"],
                                      SaveCode["Clock"], SaveCode["Energy"],
                                      SaveE,
                                      SaveB,
                                      SaveA,
                                      SaveP,
                                      SaveD,
                                      SaveC,
                                      SaveT)
                        i_save += 1
                    if self.IsPrimDeath:
                        brk = self.__brck_index["Death"]
                    if self.Verbose:
                        print(f" ### Break due to {self.__index_brck[brk]} ### ", end=' ')
                    status = "DefaultBC_" + f'{self.__index_brck[brk]}'
                    break

                if self.Verbose and (i / self.Num * 100) % 10 == 0:
                    print(f"{int(i / self.Num * 100)}%", end=' ')

            if self.Verbose:
                print("100%")
            if self.IsFirstRun:
                print(f"\t\tEvent No {self.index + 1} of {len(self.Particles)} in {timer() - st} seconds")
            if self.Verbose:
                print()
            Saves = Saves[:i_save]
            Saves[:, SaveMetric] /= self.ToMeters

            track = {"Coordinates": Saves[:, SaveCode["Coordinates"]], "Velocities": Saves[:, SaveCode["Velocities"]]}

            if SaveE:
                track["Efield"] = Saves[:, SaveCode["Efield"]]
            if SaveB:
                track["Bfield"] = Saves[:, SaveCode["Bfield"]]
            if SaveA:
                track["Angles"] = Saves[:, SaveCode["Angles"]]
            if SaveP:
                track["Path"] = Saves[:, SaveCode["Path"]]
            if SaveC:
                track["Clock"] = Saves[:, SaveCode["Clock"]]
            if SaveT:
                track["Energy"] = Saves[:, SaveCode["Energy"]]
            if SaveD:
                track["Density"] = Saves[:, SaveCode["Density"]]

            RetArr.append({"Track": track,
                           "BC": {"WOut": brk, "lon_total": lon_total},
                           "Particle": {"PDG": particle.PDG, "M": M, "Ze": particle.Z, "T0": particle.T, "Gen": Gen},
                           "Child": prod_tracks})

            # TODO refactor
            if self.Region == Regions.Magnetosphere:
                # Particles in magnetosphere (Part 1)
                if self.TrackParamsIsOn:
                    if self.Verbose:
                        print("\t\t\tGet trajectory parameters ...", end=' ')
                    TrackParams_i = Additions.GetTrackParams(self, RetArr[self.index])
                    RetArr[self.index]["Additions"] = TrackParams_i
                    if self.Verbose:
                        print("Done")

                # TODO find differences with MATLAB
                # Particles in magnetosphere (Part 2)
                if self.ParticleOriginIsOn and self.IsFirstRun:
                    if self.Verbose:
                        print("\t\t\tGet particle origin ...", end=' ')
                    origin = Additions.FindParticleOrigin(self, RetArr[self.index])
                    RetArr[self.index]["Additions"]["ParticleOrigin"] = origin
                    if self.Verbose:
                        print(origin.name)
                        print()

        return RetArr

    def __Decay(self, Gen, GenMax, T, TotTime, V_norm, Vm, particle, prod_tracks, r):
        if Gen < GenMax:
            secondary = G4Decay(particle.PDG, T)
            rotationMatrix = vecRotMat(np.array([0, 0, 1]), Vm / V_norm)
            for p in secondary:
                V_p = rotationMatrix @ p['MomentumDirection']
                r_p = r / self.ToMeters
                T_p = p['KineticEnergy']
                name_p = p["Name"]

                params = self.ParamDict.copy()
                params["Particles"] = {"Names": name_p,
                                       "T": T_p,
                                       "Center": r_p,
                                       "Radius": 0,
                                       "V0": V_p}
                params["Date"] = params["Date"] + datetime.timedelta(seconds=TotTime)
                new_process = self.__class__(**params)
                new_process.__gen = Gen + 1
                prod_tracks.append(new_process.CallOneFile()[0])

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def CheckBreak(r, r0, center, TotPath, TotTime, full_revolutions, Brck):
        radi = np.linalg.norm(r - center)
        dst2path = np.linalg.norm(r - r0) / TotPath
        cond = np.concatenate((np.array([*np.abs(r), radi, dst2path]) < Brck[:5],
                               np.array([*np.abs(r), radi, TotPath, TotTime]) > Brck[5:-1],
                               np.array([full_revolutions]) >= Brck[-1]))
        if np.any(cond):
            return True, np.where(cond)[0][0]

        return False, -1

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def Update(PathLen, Step, TotPathLen, TotTime, Vm, r):
        V_norm = np.linalg.norm(Vm)
        r_new = r + Vm * Step
        TotTime += Step
        TotPathLen += PathLen
        return V_norm, r_new, TotPathLen, TotTime

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def SaveStep(r_new, V_norm, TotPathLen, TotPathDen, TotTime, Vm, i_save, r, T, E, B, Saves,
                 CordCode, VCode, ECode, BCode, ACode, PCode, DCode, CCode, EnCode,
                 SaveE,
                 SaveB,
                 SaveA,
                 SaveP,
                 SaveD,
                 SaveC,
                 SaveT
                 ):
        Saves[i_save, CordCode] = r
        Saves[i_save, VCode] = Vm / V_norm
        if SaveE:
            Saves[i_save, ECode] = E
        if SaveB:
            Saves[i_save, BCode] = B
        if SaveA:
            Saves[i_save, ACode] = np.arctan2(np.linalg.norm(np.cross(r, r_new)), np.dot(r, r_new))
        if SaveP:
            Saves[i_save, PCode] = TotPathLen
        if SaveD:
            Saves[i_save, DCode] = TotPathDen
        if SaveC:
            Saves[i_save, CCode] = TotTime
        if SaveT:
            Saves[i_save, EnCode] = T

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def RadLossStep(Vp, Vm, Yp, Ya, M, Q, UseRadLosses, Step, ForwardTracing, c, e):
        if not UseRadLosses:
            T = M * (Yp - 1)
            return Vp, T

        acc = (Vp - Vm) / Step
        Vn = np.linalg.norm(Vp + Vm)
        Vinter = (Vp + Vm) / Vn

        acc_par = np.dot(acc, Vinter)
        acc_per = np.sqrt(np.linalg.norm(acc) ** 2 - acc_par ** 2)

        dE = Step * ((2 / (3 * 4 * np.pi * 8.854187e-12) * Q ** 2 * Ya ** 4 / c ** 3) *
                          (acc_per ** 2 + acc_par ** 2 * Ya ** 2) / e / 1e6)

        T = M * (Yp - 1) - ForwardTracing * np.abs(dE)

        V = c * np.sqrt((T + M) ** 2 - M ** 2) / (T + M)
        Vn = np.linalg.norm(Vp)

        Vm = V * Vp / Vn

        return Vm, T

    @abstractmethod
    def AlgoStep(self, T, M, q, Vm, r):
        pass
