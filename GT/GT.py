import math
import os
import numpy as np
import datetime
import json
import warnings

from numba import jit
from abc import ABC, abstractmethod
from timeit import default_timer as timer

from MagneticFields import AbsBfield
from Medium import GTGeneralMedium
from Global import Constants, Units, Regions, BreakCode, BreakIndex, SaveCode, SaveDef, BreakDef, vecRotMat
from Particle import ConvertT2R, GetAntiParticle, Flux, CRParticle
from Particle.Generators import Distributions, Spectrums
from MagneticFields.Magnetosphere import Functions, Additions
from Interaction import G4Interaction, G4Decay, SynchCounter, RadLossStep, path_geant4
warnings.simplefilter("always")


class GTSimulator(ABC):
    """
    Description

    :param Region: The region of the in which the simulation is taken place. The parameter may have values as
                   `Global.Region.Magnetosphere`, `Global.Region.Heliosphere`, `Global.Region.Galaxy`.
                   See :py:mod:`Global.regions`. See also :py:mod:`Global.regions._AbsRegion.set_params` for additional
                   energy losses.
                   Example: one mau need to take into account the adiabatic energy losses in the heliosphere.
                   In that case they call `set_params(True)`.
    :type Region: :py:mod:`Global.regions.Regions`

    :param Bfield: The magentic field object
    :type Bfield: :py:mod:`MagneticFields.magnetic_field.AbsBfield`

    :param Efield: Electrical field. Similar to :ref:`Bfield`
    :type Efield: None

    :param Medium: The medium where particles may go into an interaction. See :py:mod:`Medium`.
    :type Medium: :py:mod:`Medium.general_medium.GTGeneralMedium`

    :param Date: Date that is used to initialize the fields
    :type Date: datetime.datetime

    :param RadLosses: a `bool` flag that turns the calculations of radiation losses of the particles on.
    :type RadLosses: bool

    :param Particles: The parameter is responsible for the initial particle flux generation. It defines the initial
                      particle spectrum, distribution and chemical composition. See
                      :py:mod:`Particle.Generators.Spectrums` and :py:mod:`Particle.Generators.Distributions` for
                      available initial spectra and distributions respectively. For more information regarding flux
                      also see :py:mod:`Particle.Flux`.
    :type Particles: :py:mod:`Particle.Flux`

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

    :param Step: The time step of simulation in seconds. If `dict` one should pass:\n
        1. `UseAdaptiveStep`: `True`/`False` --- whether to use adaptive time step\n
        2. `InitialStep`: `float` --- The initial time step in seconds\n
        3. `MinLarmorRad`: `int` --- The minimal number of points during on the larmor radius\n
        4. `MaxLarmorRad`: `int` --- The maximal number of points during on the larmor radius\n
        5. `LarmorRad`:  `int` --- The fixed number of points, in case when the user needs to update time step during each step
    :type Step: float or dict

    :param Nfiles: Number of files if `int`, otherwise the `list` of file numbers (e.g. `Nfiles = [5, 10, 20]`, then
                   3 files are numerated as 5, 10, 20). If :ref:`Particles` creates a flux of `Nevents` particles then
                   the total number of particles that are going to be simulated is `Nevents`x`Nfiles`.
    :type Nfiles: int or list

    :param Output: If `None` no files are saved. Otherwise, the name of the saved *.npy* file. If :ref:`Nfiles` is
                   greater than 1. Then the names of the saved files will have the following form `"Output"_i.npy`
    :type Output: str or None

    :param Verbose: If `True` logs are printed
    :type Verbose: bool

    :param BreakCondition: If `None` no break conditions are applied. In case of a `dict` with a key corresponding to
                           the `BreakCondition` name and value corresponding to its value is passed.
                           Example: `{"Rmax": 10}`. In the example the maximal radius of the particle is 10
                           (in :py:mod:`MagneticFields` distance units). See the full list of break conditions
                           :py:mod:`Global.codes.BreakCode`.
                           If `list`, the first parameter is the `dict`, the second parameter describes the break
                           condition center, i.e. A 3d array of point (in :py:mod:`MagneticFields` distance units). It
                           represents the **(0, 0, 0)** relative to which `Rmax/Rmin, Xmax/Xmin, Ymax/Ymin, Zmax/Zmin`
                           are calculated. Default `np.array([0, 0, 0])`.
    :type BreakCondition: dict or list or None

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

                NumB0: An array of trajectory points with the minimum value of the magnetic field strength

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

    def __init__(self,
                 Bfield: None | AbsBfield = None,
                 Efield=None,
                 Region: Regions = Regions.Undefined,
                 Medium: None | GTGeneralMedium = None,
                 Date=datetime.datetime(2008, 1, 1),
                 RadLosses: bool | list = False,
                 Particles: None | Flux = None,
                 TrackParams=False,
                 ParticleOrigin=False,
                 IsFirstRun=True,
                 ForwardTrck=None,
                 Save: int | list = 1,
                 Num: int = 1e6,
                 Step: float = 1,
                 Nfiles=1,
                 Output=None,
                 Verbose=False,
                 BreakCondition: None | dict = None,
                 UseDecay=False,
                 InteractNUC: None | dict = None,
                 ):

        self.ParamDict = locals().copy()
        del self.ParamDict['self']

        self.Verbose = Verbose
        if self.Verbose:
            print("Creating simulator object...")

        self.Date = Date
        if self.Verbose:
            print(f"\tDate: {self.Date}")
            print()

        self.StepParams = Step
        self.Step = None
        self.UseAdaptiveStep = False
        self.__SetStep(Step)

        self.Num = int(Num)
        if self.Verbose:
            print(f"\tNumber of steps: {self.Num}")
            print()

        self.__SetUseRadLosses(RadLosses)
        if self.Verbose:
            print()

        self.__SetRegion(Region)

        self.ParticleOrigin = ParticleOrigin
        self.ParticleOriginIsOn = False
        if self.ParticleOrigin:
            self.ParticleOriginIsOn = True
            TrackParams = True

        self.TrackParamsIsOn = False
        self.TrackParams = self.Region.value.SaveAdd
        self.__SetAdditions(TrackParams)

        if self.TrackParamsIsOn:
            if not isinstance(Save, list):
                Save = [Save, {"Bfield": True}]
            elif "Bfield" not in Save[1]:
                Save[1] = Save[1] | {"Bfield": True}

        self.IsFirstRun = IsFirstRun
        self.Nfiles = 1 if Nfiles is None or Nfiles == 0 else Nfiles
        self.Output = Output
        self.Npts = 2
        self.Save = SaveDef.copy()
        self.SaveCode = dict([(key, SaveCode[key][1]) for key in SaveCode.keys()])
        self.SaveColumnLen = 17
        if self.Verbose:
            print(f"\tNumber of files: {self.Nfiles}")
            print(f"\tOutput file name: {self.Output}_num.npy")
        self.__SetSave(Save)
        if self.Verbose:
            print()

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

        self.__brck_index = BreakCode.copy()
        self.__brck_index.pop("Loop")
        self.__index_brck = BreakIndex.copy()
        self.__brck_arr = BreakDef.copy()
        self.__SetBrck(BreakCondition)

        self.index = 0
        if self.Verbose:
            print("Simulator created!\n")

    def __SetStep(self, Step):
        if isinstance(Step, (int, float)):
            self.Step = Step
            if self.Verbose:
                print(f"\tTime step: {self.Step}")
        elif isinstance(Step, dict):
            self.UseAdaptiveStep = Step.get("UseAdaptiveStep", False)
            self.Step = Step.get("InitialStep", 1)
            N = Step.get("LarmorRad", None)
            if N is not None:
                self.N1 = self.N2 = N
            else:
                self.N1 = Step.get("MinLarmorRad", 600)
                self.N2 = Step.get("MaxLarmorRad", 600)
            assert isinstance(self.UseAdaptiveStep, bool)
            assert isinstance(self.Step, (int, float))
            assert isinstance(self.N1, int) and isinstance(self.N2, int)
            assert self.N1<=self.N2
        else:
            raise Exception("Step should be numeric or dict")

        if self.Verbose:
            if not self.UseAdaptiveStep:
                print(f"\tTime step: {self.Step}")
            else:
                print(f"\tUsing adaptive time step: True")
                print(f"\tInitial time step: {self.Step}")
                if N is None:
                    print(f"\tMinimal number of steps in larmor radius: {self.N1}")
                    print(f"\tMaximal number of steps in larmor radius: {self.N2}")
                else:
                    print(f"\tNumber of steps in larmor radius: {N}")

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
        if UseInteractNUC is not None and not os.path.exists(f"{path_geant4}/bin/geant4.sh"):
            raise ValueError("Geant4 setup script was not found. Please, check path_geant4 variable in the settings file.")
        self.InteractNUC = UseInteractNUC
        if self.InteractNUC is not None and 'l' in self.InteractNUC.get("ExcludeParticleList", []):
            self.InteractNUC['ExcludeParticleList'].extend([11, 12, 13, 14, 15, 16, 17, 18,
                                                            -11, -12, -13, -14, -15, -16, -17, -18])
        self.IntPathDen = 10  # g/cm2
        if self.Verbose:
            print(f"\tDecay: {self.UseDecay}")
            print(f"\tNuclear Interactions: {self.InteractNUC}")

    def __SetBrck(self, Brck):
        center = np.array([0, 0, 0])
        if Brck is not None:
            if isinstance(Brck, list):
                center = Brck[1]
                assert isinstance(center, np.ndarray) and center.shape == (3,)
                Brck = Brck[0]
                assert isinstance(Brck, dict)
            for key in Brck.keys():
                self.__brck_arr[self.__brck_index[key]] = Brck[key]
        if self.Verbose:
            print("\tBreak Conditions: ")
            for key in self.__brck_index.keys():
                print(f"\t\t{key}: {self.__brck_arr[self.__brck_index[key]]}")
            print(f"\tBC center: {center}")
        self.BCcenter = center

    def __SetMedium(self, medium):
        if self.Verbose:
            print("\tMedium: ", end='')
        if medium is not None:
            self.Medium = medium
            if self.Verbose:
                print(str(self.Medium))
        else:
            if self.Verbose:
                print(None)

    def __SetFlux(self, flux, forward_trck):
        if self.Verbose:
            print("\tFlux: ", end='')
        assert flux is not None
        self.Particles = flux

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
            self.Efield = Efield
            if self.Verbose:
                print(str(self.Efield))
        else:
            if self.Verbose:
                print(None)

        if self.Verbose:
            print("\tMagnetic field: ", end='')
        if Bfield is not None:
            self.Bfield = Bfield
            if self.Verbose:
                print(str(self.Bfield))
        else:
            if self.Verbose:
                print(None)

    def __SetRegion(self, Region):
        self.Region = Region

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
        if isinstance(Save, list):
            for saves in Save[1].keys():
                self.Save[saves] = Save[1][saves]

        sorted_keys = sorted(SaveCode, key = lambda x: SaveCode[x][0])
        for i, key in enumerate(sorted_keys):
            if not self.Save[key]:
                val = SaveCode[key][1]
                num = 0
                self.SaveCode[key] = None
                if isinstance(val, int):
                    num = 1
                elif isinstance(val, slice):
                    num = val.stop - val.start

                self.SaveColumnLen -= num

                for j in range(i + 1, len(sorted_keys)):
                    val_ = self.SaveCode[sorted_keys[j]]
                    if isinstance(val_, int):
                        val_ -= num
                    elif isinstance(val_, slice):
                        val_ = np.s_[val_.start - num:val_.stop - num:1]
                    self.SaveCode[sorted_keys[j]] = val_

        if self.Verbose:
            for saves in self.Save.keys():
                print(f"\t\t{saves}: {self.Save[saves]}")

    def __call__(self):
        Track = []
        if self.Verbose:
            print("Launching simulation...")
        file_nums = np.arange(self.Nfiles) if isinstance(self.Nfiles, int) else self.Nfiles
        for (idx, i) in enumerate(file_nums):
            if self.IsFirstRun:
                print(f"\tFile number {i}. No {idx+1} file out of {len(file_nums)}")
            if self.Output is not None:
                file = self.Output.split(os.sep)
                folder = os.sep.join(file[:-1])
                if len(file) != 1 and not os.path.isdir(folder):
                    os.mkdir(folder)
                def custom_serializer(obj):
                    if isinstance(obj, (AbsBfield, GTGeneralMedium)):
                        lines = [el.strip() for el in str(obj).strip().split('\n')]
                        return [lines[0], dict([el.split(': ') for el in lines[1:]])]
                    if isinstance(obj, Flux):
                        return dict([el.strip().split(': ') for el in str(obj).strip().split('\n')])
                    if isinstance(obj, Regions):
                        return str(obj)
                    if isinstance(obj, datetime.datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")
                def custom_serializer_safe(obj):
                    return str(obj)
                with open(f'{self.Output}_params.json', 'w') as file:
                    try:
                        json.dump(self.ParamDict, file, default=custom_serializer, indent=4)
                    except:
                        json.dump(self.ParamDict, file, default=custom_serializer_safe, indent=4)

            RetArr = self.CallOneFile()

            if self.Output is not None:
                if self.Nfiles == 1:
                    np.save(f"{self.Output}.npy", RetArr)
                else:
                    np.save(f"{self.Output}_{i}.npy", RetArr)
                if self.Verbose:
                    print("\tFile saved!")
                RetArr.clear()
            else:
                Track.append(RetArr)

        if self.Verbose:
            print("Simulation completed!")
        if self.Output is None:
            return Track

    def CallOneFile(self):
        self.Particles.generate()
        RetArr = []
        status = "Done"

        SaveR = self.Save["Coordinates"]
        SaveV = self.Save["Velocities"]
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
                LocalDen, nLocal, LocalPathDen = 0, 0, 0
                LocalChemComp = np.zeros(len(self.Medium.get_element_list()))
                LocalPathDenVector = np.empty(0)
                LocalCoordinate = np.empty([0, 3])
                LocalVelocity = np.empty([0, 3])
            lon_total, lon_prev, full_revolutions = np.array([[0.]]), np.array([[0.]]), 0
            particle = self.Particles[self.index]
            Saves = []
            BrckArr = self.__brck_arr
            BCcenter = self.BCcenter
            tau = particle.tau
            rnd_dec = 0
            self.IsPrimDeath = False
            prod_tracks = []
            if self.UseDecay:
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
            m = M*Units.MeV2kg
            T = particle.T

            V_normalized = np.array(particle.velocities)  # unit vector of velocity (beta vector)
            V_norm = Constants.c * np.sqrt(particle.E ** 2 - M ** 2) / particle.E  # scalar speed [m/s]
            Vm = V_norm * V_normalized  # vector of velocity [m/s]

            r0 = np.array(particle.coordinates)
            r = np.array(particle.coordinates)
            r_old = r

            B = np.array(self.Bfield.GetBfield(*r)) if self.Bfield is not None else np.zeros(3)
            E = np.array(self.Efield.GetEfield(*r)) if self.Efield is not None else np.zeros(3)

            Step = self.Step

            if Q == 0:
                Step *= 1e2

            if self.Verbose:
                print(f"\t\t\tParticle: {particle.Name} (M = {M} [MeV], Z = {self.Particles[self.index].Z})")
                print(f"\t\t\tEnergy: {T} [MeV], Rigidity: "
                      f"{ConvertT2R(T, M, particle.A, particle.Z) / 1000 if particle.Z != 0 else np.inf} [GV]")
                print(f"\t\t\tCoordinates: {r} [m]")
                print(f"\t\t\tVelocity: {V_normalized}")
                print(f"\t\t\tbeta: {V_norm / Constants.c}")
                print(f"\t\t\tbeta*dt: {V_norm * Step / 1000} [km] / "
                      f"{V_norm * Step} [m]")

            # Calculation of EAS for magnetosphere
            self.Region.value.do_before_loop(self, Gen, prod_tracks)

            q = Step * Q / 2 / m if M != 0 else 0
            brk = BreakCode["Loop"]

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
                if i % Nsave == 0 or i == Num - 1 or i_save == 0:
                    sv = np.zeros(self.SaveColumnLen)
                    self.SaveStep(r_old, V_norm, TotPathLen, TotPathDen, TotTime, Vm, i_save, r, T, E, B, Saves,
                                  self.SaveColumnLen,
                                  self.SaveCode["Coordinates"], self.SaveCode["Velocities"], self.SaveCode["Efield"],
                                  self.SaveCode["Bfield"], self.SaveCode["Angles"], self.SaveCode["Path"],
                                  self.SaveCode["Density"], self.SaveCode["Clock"], self.SaveCode["Energy"],
                                  SaveR, SaveV, SaveE, SaveB, SaveA, SaveP, SaveD, SaveC, SaveT)
                    i_save += 1

                if self.UseAdaptiveStep:
                    Step = self.AdaptStep(Q, m, B, Vm, T, M, Step, self.N1, self.N2)
                    if i == 0:
                        self.Step = Step
                    q = Step * Q / 2 / m

                PathLen = V_norm * Step

                Vp, Yp, Ya = self.AlgoStep(T, M, q, Vm, r, B, E)

                if self.UseRadLosses[1]:
                    synch_record.add_iteration(T, B, Vm, Step)
                if self.UseRadLosses[0]:
                    Vm, T, new_photons, synch_record = RadLossStep.MakeRadLossStep(Vp, Vm, Yp, Ya, M, Q, r,
                                                                                   Step, self.ForwardTracing,
                                                                                   self.UseRadLosses[1:], particle, Gen,
                                                                                   Constants,
                                                                                   synch_record)
                    prod_tracks.extend(new_photons)
                elif M > 0:
                    T = M * (Yp - 1)
                    Vm = Vp

                if UseAdditionalEnergyLosses:
                    Vm, T = self.Region.value.AdditionalEnergyLosses(r, Vm, T, M, Step, self.ForwardTracing,
                                                                     Constants.c)
                r_old = r
                V_norm, r, TotPathLen, TotTime = self.Update(PathLen, Step, TotPathLen, TotTime, Vm, r)

                # Medium
                if self.Medium is not None:
                    self.Medium.calculate_model(*r)
                    Den = self.Medium.get_density()  # kg/m3
                    PathDen = (Den * 1e-3) * (PathLen * 1e2)  # g/cm2
                    TotPathDen += PathDen  # g/cm2
                    if self.InteractNUC is not None and Den > 0:
                        LocalDen += Den
                        LocalChemComp += self.Medium.get_element_abundance()
                        nLocal += 1
                        LocalPathDen += PathDen
                        LocalPathDenVector = np.append(LocalPathDenVector, LocalPathDen)
                        LocalCoordinate = np.append(LocalCoordinate, r[None, :], axis=0)
                        LocalVelocity = np.append(LocalVelocity, Vm[None, :], axis=0)

                # Decay
                if self.UseDecay and not self.IsPrimDeath:
                    lifetime = tau * (T / M + 1) if M > 0 else np.inf
                    if rnd_dec > np.exp(-TotTime / lifetime):
                        self.__Decay(Gen, GenMax, T, TotTime, V_norm, Vm, particle, prod_tracks, r)
                        self.IsPrimDeath = True

                # Nuclear Interaction
                if self.InteractNUC is not None and LocalPathDen > self.IntPathDen and not self.IsPrimDeath:
                    # Construct Rotation Matrix & Save velocity before possible interaction
                    rotationMatrix = vecRotMat(np.array([0, 0, 1]), Vm / V_norm)
                    primary, secondary = G4Interaction(particle.PDG, T, LocalPathDen, (LocalDen * 1e-3) / nLocal,
                                                       self.Medium.get_element_list(), LocalChemComp / nLocal)
                    T = primary['KineticEnergy']
                    if T > 0 and T > 1:  # Cut particles with T < 1 MeV
                        # Only ionization losses
                        V_norm = Constants.c * np.sqrt(1 - (M / (T + M)) ** 2)
                        Vm = V_norm * rotationMatrix @ primary['MomentumDirection']
                        LocalDen, nLocal, LocalPathDen = 0, 0, 0
                        LocalChemComp = np.zeros(len(self.Medium.get_element_list()))
                        LocalPathDenVector = np.empty(0)
                        LocalCoordinate = np.empty([0, 3])
                        LocalVelocity = np.empty([0, 3])
                    else:
                        # Death due to ionization losses or nuclear interaction
                        self.IsPrimDeath = True
                        if secondary.size > 0 and Gen < GenMax:
                            if self.Verbose:
                                print(
                                    f"Nuclear interaction ~ {primary['LastProcess']} ~ {secondary.size} secondaries ~ {np.sum(secondary['KineticEnergy'])} MeV")
                                print(secondary)
                            # Coordinates of interaction point in XYZ
                            path_den_cylinder = (np.linalg.norm(primary['Position']) * 1e2) * (
                                    LocalDen * 1e-3 / nLocal)  # Path in cylinder [g/cm2]
                            r_interaction = LocalCoordinate[np.argmax(LocalPathDenVector > path_den_cylinder), :]
                            v_interaction = LocalVelocity[np.argmax(LocalPathDenVector > path_den_cylinder), :]
                            rotationMatrix = vecRotMat(np.array([0, 0, 1]), v_interaction / np.linalg.norm(v_interaction))
                            for p in secondary:
                                V_p = rotationMatrix @ p['MomentumDirection']
                                T_p = p['KineticEnergy']
                                PDGcode_p = p["PDGcode"]
                                # Try to find a particle (TODO: REMOVE IN THE FUTURE)
                                try:
                                    name_p = CRParticle(PDG=PDGcode_p, Name=None).Name
                                except:
                                    warnings.warn(
                                        f"Particle with code {PDGcode_p} was not found. Calculation is skipped.")
                                    continue
                                # Parameters for recursive call of GT
                                params = self.ParamDict.copy()
                                params["Date"] += datetime.timedelta(seconds=TotTime)
                                params["Particles"] = Flux(
                                    Distribution=Distributions.UserInput(R0=r_interaction, V0=V_p),
                                    Spectrum=Spectrums.UserInput(energy=T_p),
                                    Names=name_p
                                )
                                if PDGcode_p in self.InteractNUC.get("ExcludeParticleList",
                                                                     []) or T_p < self.InteractNUC.get("Emin", 0):
                                    params["Num"] = 1
                                    params["UseDecay"] = False
                                    params["InteractNUC"] = None
                                if PDGcode_p in [12, 14, 16, 18, -12, -14, -16, -18]:
                                    params["Medium"] = None
                                    params["InteractNUC"] = None
                                new_process = self.__class__(**params)
                                new_process.__gen = Gen + 1
                                prod_tracks.append(new_process.CallOneFile()[0])

                B = np.array(self.Bfield.GetBfield(*r)) if self.Bfield is not None else np.zeros(3)
                E = np.array(self.Efield.GetEfield(*r)) if self.Efield is not None else np.zeros(3)

                # TODO the code is region specific
                # Full revolution
                if (self.ParticleOriginIsOn or self.__brck_arr[self.__brck_index["MaxRev"]] != BreakDef[-1]) and \
                        self.Region == Regions.Magnetosphere:
                    a_, b_, _ = Functions.transformations.geo2mag_eccentric(r[0], r[1], r[2], 1, self.Bfield.g,
                                                                            self.Bfield.h)
                    lon_total, lon_prev, full_revolutions = Additions.AddLon(lon_total, lon_prev, full_revolutions, i,
                                                                             a_, b_)

                brck = self.CheckBreak(r, r0, BCcenter, TotPathLen, TotTime, full_revolutions, BrckArr)
                brk = brck[1]
                if brck[0] or self.IsPrimDeath:
                    if brk != -1:
                        self.SaveStep(r_old, V_norm, TotPathLen, TotPathDen, TotTime, Vm, i_save, r, T, E, B, Saves,
                                      self.SaveColumnLen,
                                      self.SaveCode["Coordinates"], self.SaveCode["Velocities"], self.SaveCode["Efield"],
                                      self.SaveCode["Bfield"], self.SaveCode["Angles"], self.SaveCode["Path"],
                                      self.SaveCode["Density"], self.SaveCode["Clock"], self.SaveCode["Energy"],
                                      SaveR, SaveV, SaveE, SaveB, SaveA, SaveP, SaveD, SaveC, SaveT)
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
            Saves = np.array(Saves)

            track = {}
            if SaveR:
                track['Coordinates'] = Saves[:, self.SaveCode["Coordinates"]]
            if SaveV:
                track["Velocities"] = Saves[:, self.SaveCode["Velocities"]]
            if SaveE:
                track["Efield"] = Saves[:, self.SaveCode["Efield"]]
            if SaveB:
                track["Bfield"] = Saves[:, self.SaveCode["Bfield"]]
            if SaveA:
                track["Angles"] = Saves[:, self.SaveCode["Angles"]]
            if SaveP:
                track["Path"] = Saves[:, self.SaveCode["Path"]]
            if SaveC:
                track["Clock"] = Saves[:, self.SaveCode["Clock"]]
            if SaveT:
                track["Energy"] = Saves[:, self.SaveCode["Energy"]]
            if SaveD:
                track["Density"] = Saves[:, self.SaveCode["Density"]]

            RetArr.append({"Track": track,
                           "BC": {"WOut": brk, "lon_total": lon_total},
                           "Particle": {"PDG": particle.PDG, "M": M, "Ze": particle.Z, "Gen": Gen,
                                        "R0": particle.coordinates, "V0": particle.velocities, "T0": particle.T},
                           "Child": prod_tracks})

            # TODO refactor
            if self.Region == Regions.Magnetosphere:
                # Particles in magnetosphere (Part 1)
                if self.TrackParamsIsOn:
                    if self.Verbose:
                        print("\t\t\tCalculating additional parameters ...", end=' ')
                    TrackParams_i = Additions.GetTrackParams(self, RetArr[self.index])
                    RetArr[self.index]["Additions"] = TrackParams_i
                    if self.Verbose:
                        print("Done")

                # TODO find differences with MATLAB
                # Particles in magnetosphere (Part 2)
                if self.ParticleOriginIsOn and self.IsFirstRun:
                    if self.Verbose:
                        print("\t\t\tFinding particle origin ...", end=' ')
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
                r_p = r
                T_p = p['KineticEnergy']
                PDGcode_p = p["PDGcode"]
                # Try to find a particle (TODO: REMOVE IN THE FUTURE)
                try:
                    name_p = CRParticle(PDG=PDGcode_p, Name=None).Name
                except:
                    warnings.warn(f"Particle with code {PDGcode_p} was not found. Calculation is skipped.")
                    continue
                params = self.ParamDict.copy()
                params["Particles"] = Flux(
                    Distribution=Distributions.UserInput(R0=r_p, V0=V_p),
                    Spectrum=Spectrums.UserInput(energy=T_p),
                    Names=name_p
                )
                params["Date"] = params["Date"] + datetime.timedelta(seconds=TotTime)
                if PDGcode_p in [12, 14, 16, 18, -12, -14, -16, -18]:
                    params["Medium"] = None
                    params["InteractNUC"] = None
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
    # @jit(fastmath=True, nopython=True)
    def SaveStep(r_old, V_norm, TotPathLen, TotPathDen, TotTime, Vm, i_save, r, T, E, B, Saves, ColLen,
                 RCode, VCode, ECode, BCode, ACode, PCode, DCode, CCode, TCode,
                 SaveR, SaveV, SaveE, SaveB, SaveA, SaveP, SaveD, SaveC, SaveT):
        sv = np.zeros(ColLen)
        if SaveR:
            sv[RCode] = r
        if SaveV:
            sv[VCode] = Vm / V_norm
        if SaveE:
            sv[ECode] = E
        if SaveB:
            sv[BCode] = B
        if SaveA:
            sv[ACode] = np.arctan2(np.linalg.norm(np.cross(r_old, r)), np.dot(r, r_old))
        if SaveP:
            sv[PCode] = TotPathLen
        if SaveD:
            sv[DCode] = TotPathDen
        if SaveC:
            sv[CCode] = TotTime
        if SaveT:
            sv[TCode] = T

        Saves.append(sv)

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

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def AdaptStep(q, m, B, V, T, M, dt, N1, N2):
        Y = T / M + 1
        B_n = np.linalg.norm(B)
        cos_theta = B @ V / (np.linalg.norm(V) * B_n)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        T = Y * m * sin_theta / (np.abs(q) * B_n)
        if N1 <= T / dt <= N2:
            return dt

        return T / np.sqrt(N1*N2)

    @abstractmethod
    def AlgoStep(self, T, M, q, Vm, r, H, E):
        pass
