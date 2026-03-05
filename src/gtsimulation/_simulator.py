import datetime
import json
import logging
import math
import os
import sys
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np
from numba import njit

from gtsimulation import functions
from gtsimulation.ElectricFields import GeneralFieldE
from gtsimulation.Global import Constants, Units, Regions, BreakCode, BreakIndex, SaveCode, SaveDef, BreakDef, vecRotMat
from gtsimulation.Interaction import NuclearInteraction, G4Decay, SynchCounter, RadLossStep
from gtsimulation.MagneticFields import AbsBfield
from gtsimulation.MagneticFields.Magnetosphere import Functions, Additions
from gtsimulation.Medium import GTGeneralMedium
from gtsimulation.Particle import ConvertT2R, GetAntiParticle, Flux
from gtsimulation.Particle.Generators import Distributions, Spectrums


class GTSimulator(ABC):
    """
    Main simulation class for particle trajectory calculation in various space environments.

    This abstract base class provides the core functionality for simulating particle
    trajectories, taking into account electromagnetic fields, particle interactions,
    and various physical processes.

    Parameters
    ----------
    Particles : :py:class:`~gtsimulation.Particle.Flux`
        Particle flux generator defining initial spectrum, distribution and composition.
        See :py:mod:`gtsimulation.Particle`.

    Num : int
        Maximum number of simulation steps.

    Step : float or dict
        Time step configuration. If float: fixed time step in seconds.
        If dict: adaptive step configuration with keys:

        * ``UseAdaptiveStep``: bool – Enable adaptive stepping (default: False).
        * ``InitialStep``: float – Initial time step in seconds (used only if adaptive).
        * ``MinLarmorRad``: int – Minimum number of Larmor radii per step.
        * ``MaxLarmorRad``: int – The minimal number of points per Larmor radius.
        * ``LarmorRad``: int – The fixed number of points per Larmor radius (used when a fixed resolution is desired).

    Bfield : :py:class:`~gtsimulation.MagneticFields.AbsBfield` or None, optional
        Magnetic field model object. If None, no magnetic field is applied.
        Default is None.

    Efield : :py:class:`~gtsimulation.ElectricFields.GeneralFieldE` or None, optional
        Electric field model object. If None, no electric field is applied.
        Default is None.

    Medium : :py:class:`~gtsimulation.Medium.GTGeneralMedium` or None, optional
        Propagation medium for particles. Defines density and composition of the environment
        through which particles travel. Required when nuclear interactions are enabled
        (via `InteractNUC` parameter), but optional for simulations without interactions.
        See :py:mod:`gtsimulation.Medium`.
        Default is None.

    Region : :py:class:`~gtsimulation.Global.regions.Regions`, optional
        Simulation region specifying the physical environment.
        Available options:
        :py:attr:`~gtsimulation.Global.regions.Regions.Magnetosphere`,
        :py:attr:`~gtsimulation.Global.regions.Regions.Heliosphere`,
        :py:attr:`~gtsimulation.Global.regions.Regions.Galaxy`.
        See :py:mod:`gtsimulation.Global.regions`.
        Default is `Regions.Undefined`.

    BreakCondition : dict or list or None, optional
        Simulation termination conditions. If dict: {condition: value}.
        If list format: [conditions_dict, center_point_array].
        Available conditions: see :py:data:`~gtsimulation.Global.codes.BreakCode`.
        Default is None.

    UseDecay : bool, optional
        Enable particle decay processes.
        Default is False.

    InteractNUC : :py:class:`~gtsimulation.Interaction.NuclearInteraction` or None, optional
        Nuclear interaction module. Requires Medium to be set.
        Default is None.

    RadLosses : bool or list, optional
        Radiation losses configuration. If False, radiation losses are disabled.
        If list format: [True, {"Photons": True/False, "MinE": float, "MaxE": float}].
        Default is False.

    Date : datetime.datetime, optional
        Date for field model initialization.
        Default is datetime.datetime(2008, 1, 1).

    Save : int or list, optional
        Save interval configuration. If int N, save every N steps.
        If list format: [N, {"Clock": True, ...}] to save additional parameters.
        See :py:data:`~gtsimulation.Global.codes.SaveCode`.
        Default is 1.

    Nfiles : int or list, optional
        Number of output files. If int: create N files.
        If list: create files with specified indices.
        Default is 1.

    Output : str or None, optional
        Output file base name. If None, results are not saved to disk.
        Default is None.

    Verbose : int, optional
        Verbosity level: 0 (no output), 1 (short output), 2 (verbose output).
        Default is 1.

    ForwardTrck : {1, -1} or None, optional
        Tracing direction: 1 for forward, -1 for backward. If None, determined from Particles.
        Default is None.

    TrackParams : bool or dict, optional
        Additional parameter tracking. If True, all available parameters are tracked.
        If dict, specify which parameters to track:
        {'Invariants': True, 'GuidingCenter': True, etc.}.
        See :py:attr:`~gtsimulation.Global.regions._AbsRegion.SaveAdd`.
        Default is False.

    ParticleOrigin : bool, optional
        Enable particle origin calculation through backtracing.
        Default is False.

    IsFirstRun : bool, optional
        Flag indicating first simulation run. Affects logging and some calculations.
        Default is True.

    Returns
    -------
    result : list or None
        Results of simulation.

        * If `Output` is None:
            Returns a list of simulation results. The structure depends on `Nfiles`:

            - `Nfiles = 1` — a flat list of dictionaries, one per simulated particle
              (the number of particles is determined by the `Particles` argument).
            - `Nfiles > 1` — a list of lists. Each inner list corresponds to one output file
              and contains dictionaries for the particles processed in that file.
        * If `Output` is not None:
            Results are saved to disk (NumPy binary files) and the method returns `None`.

    Notes
    -----
    For detailed information about the structure of the particle data dictionaries,
    see the **Simulation output** section in the online documentation.

    See Also
    --------
    :py:class:`gtsimulation.Particle.Flux` : Particle flux generation
    :py:data:`gtsimulation.Global.codes.SaveCode` : Available save parameters
    :py:data:`gtsimulation.Global.codes.BreakCode` : Available break conditions
    :py:class:`gtsimulation.MagneticFields` : Magnetic field module
    :py:class:`gtsimulation.Medium` : Medium module
    """

    def __init__(
            self,
            Particles: Flux,
            Num: int,
            Step: float,
            Bfield: AbsBfield | None = None,
            Efield: GeneralFieldE | None = None,
            Medium: GTGeneralMedium | None = None,
            Region: Regions = Regions.Undefined,
            BreakCondition: dict | list | None = None,
            UseDecay: bool = False,
            InteractNUC: NuclearInteraction | None = None,
            RadLosses: bool | list = False,
            Date: datetime.datetime = datetime.datetime(2008, 1, 1),
            Save: int | list = 1,
            Nfiles: int | list = 1,
            Output: str | None = None,
            Verbose: int = 1,
            ForwardTrck=None,
            TrackParams=False,
            ParticleOrigin=False,
            IsFirstRun=True,
    ):
        self.ParamDict = locals().copy()
        del self.ParamDict['self']

        self.logger = logging.getLogger(__name__)
        if Verbose < 1:
            self.logger.setLevel(logging.WARNING)
        elif Verbose == 1:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)

        if Verbose > 0 and not self.logger.hasHandlers():
            h = logging.StreamHandler(stream=sys.stdout)
            h.setLevel(self.logger.level)
            h.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(h)
            self.logger.propagate = False

        self.logger.debug("Creating simulator object...")
        self.Date = Date
        self.logger.debug("Date: %s", self.Date)

        self.StepParams = Step
        self.Step = None
        self.UseAdaptiveStep = False
        self.__SetStep(Step)

        self.Num = int(Num)
        self.logger.debug("Number of steps: %d", self.Num)

        self.__SetUseRadLosses(RadLosses)

        self.Region = Region
        self.logger.debug("Region: %s", self.Region.name)
        self.logger.debug("%s", self.Region.value.ret_str())

        self.ParticleOrigin = ParticleOrigin
        self.ParticleOriginIsOn = bool(self.ParticleOrigin)
        if self.ParticleOrigin:
            TrackParams = True

        self.TrackParamsIsOn = False
        self.TrackParams = self.Region.value.SaveAdd
        self.__SetAdditions(TrackParams, Save)

        self.IsFirstRun = IsFirstRun
        self.Nfiles = 1 if Nfiles is None or Nfiles == 0 else Nfiles
        self.Output = Output
        self.Npts = 2
        self.Save = SaveDef.copy()
        self.SaveCode = dict([(key, SaveCode[key][1]) for key in SaveCode.keys()])
        self.SaveColumnLen = 22
        self.logger.debug("Number of files: %s", self.Nfiles)
        self.logger.debug("Output file name: %s_num.npy", self.Output)

        if BreakCondition is not None and hasattr(BreakCondition, 'keys') and 'MaxRev' in BreakCondition.keys():
            if not isinstance(Save, list):
                Save = [Save, {'GuidingCenter': True, 'PitchAngles': True}]
            else:
                Save[1] = Save[1] | {'GuidingCenter': True, 'PitchAngles': True}

        self.__SetSave(Save)

        self.Efield = Efield
        self.logger.debug("Electric field: %s", self.Efield)

        self.Bfield = Bfield
        self.logger.debug("Magnetic field: %s", self.Bfield)

        self.Medium = Medium
        self.logger.debug("Medium: %s", self.Medium)

        self.UseDecay = UseDecay
        self.logger.debug("Decay: %s", self.UseDecay)

        if self.Medium is None and InteractNUC is not None:
            raise ValueError('Nuclear Interaction is enabled but Medium is not set')
        self.nuclear_interaction = InteractNUC
        self.logger.debug("Nuclear Interactions: %s", self.nuclear_interaction)

        self.__gen = 1
        # self.UseDecay = False
        # self.nuclear_interaction = None
        # self.__set_nuclear_interaction(UseDecay, InteractNUC)

        self.Particles = None
        self.ForwardTracing = 1
        self.__SetFlux(Particles, ForwardTrck)

        self.__brck_index = BreakCode.copy()
        self.__brck_index.pop("Loop")
        self.__index_brck = BreakIndex.copy()
        self.__brck_arr = BreakDef.copy()
        self.__set_break_condition(BreakCondition)

        self.index = 0
        self.logger.debug("Simulator object created!\n")

    def __SetStep(self, Step):
        if isinstance(Step, (int, float)):
            self.Step = Step
            self.logger.debug("Time step: %s seconds", self.Step)
        elif isinstance(Step, dict):
            self.UseAdaptiveStep = Step.get("UseAdaptiveStep", False)
            self.Step = Step.get("InitialStep", 1)
            self.logger.debug("Using adaptive time step: %s", self.UseAdaptiveStep)
            self.logger.debug("Initial time step: %f seconds", self.Step)

            N = Step.get("LarmorRad", None)
            if N is not None:
                self.N1 = self.N2 = N
                self.logger.debug("Steps per Larmor radius: %d", N)
            else:
                self.N1 = Step.get("MinLarmorRad", 600)
                self.N2 = Step.get("MaxLarmorRad", 600)
                self.logger.debug("Min steps per Larmor radius: %d", self.N1)
                self.logger.debug("Max steps per Larmor radius: %d", self.N2)

            assert isinstance(self.UseAdaptiveStep, bool)
            assert isinstance(self.Step, (int, float))
            assert isinstance(self.N1, int) and isinstance(self.N2, int)
            assert self.N1 <= self.N2
        else:
            raise Exception("Step should be numeric or dict")

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

        self.logger.debug("Radiation Losses: %s", self.UseRadLosses[0])
        self.logger.debug("Synchrotron Emission: %s", self.UseRadLosses[1])

    def __SetAdditions(self, TrackParams, Save):
        # Change save settings due to dependencies
        if isinstance(TrackParams, dict):
            if "GuidingCenter" in TrackParams.keys() and TrackParams["GuidingCenter"]:
                TrackParams["PitchAngles"] = True

                if not isinstance(Save, list):
                    Save = [Save, {"GuidingCenter": True, "PitchAngles": True}]
                else:
                    Save[1] = Save[1] | {"GuidingCenter": True, "PitchAngles": True}

            if "Lshell" in TrackParams.keys() and TrackParams["Lshell"]:
                TrackParams["Invariants"] = True
            if "Invariants" in TrackParams.keys() and TrackParams["Invariants"]:
                TrackParams["MirrorPoints"] = True
            if "MirrorPoints" in TrackParams.keys() and TrackParams["MirrorPoints"]:
                TrackParams["PitchAngles"] = True

                if not isinstance(Save, list):
                    Save = [Save, {"PitchAngles": True}]
                else:
                    Save[1] = Save[1] | {"PitchAngles": True}

        if isinstance(TrackParams, bool):
            self.TrackParamsIsOn = TrackParams
            if self.TrackParamsIsOn:
                self.TrackParams.update((key, True) for key in self.TrackParams)
        elif isinstance(TrackParams, dict):
            self.TrackParamsIsOn = True
            for add in TrackParams.keys():
                assert add in self.TrackParams.keys(), f'No such option as "{add}" is allowed'
                self.TrackParams[add] = TrackParams[add]

        if self.TrackParamsIsOn:
            if not isinstance(Save, list):
                Save = [Save, {"Bfield": True}]
            else:
                Save[1] = Save[1] | {"Bfield": True}

    # def __set_nuclear_interaction(self, UseDecay, UseInteractNUC):
    #     self.UseDecay = UseDecay
    #     if self.Medium is None and UseInteractNUC is not None:
    #         raise ValueError('Nuclear Interaction is enabled but Medium is not set')
    #     self.nuclear_interaction = UseInteractNUC
    #     if self.nuclear_interaction is not None and 'l' in self.nuclear_interaction.get("ExcludeParticleList", []):
    #         self.nuclear_interaction['ExcludeParticleList'].extend([11, 12, 13, 14, 15, 16, 17, 18,
    #                                                                 -11, -12, -13, -14, -15, -16, -17, -18])
    #     self.logger.debug("Decay: %s", self.UseDecay)
    #     self.logger.debug("Nuclear Interactions: %s", self.nuclear_interaction)

    def __set_break_condition(self, Brck):
        center = np.array([0, 0, 0])
        if Brck is not None:
            if isinstance(Brck, list):
                center = Brck[1]
                assert isinstance(center, np.ndarray) and center.shape == (3,)
                Brck = Brck[0]
            assert isinstance(Brck, dict)
            for key in Brck.keys():
                self.__brck_arr[self.__brck_index[key]] = Brck[key]
        self.logger.debug("Break Conditions:")
        for key in self.__brck_index.keys():
            self.logger.debug("\t%s: %s", key, self.__brck_arr[self.__brck_index[key]])
        self.logger.debug("BC center: %s", center)
        self.BCcenter = center

    def __SetFlux(self, flux, forward_trck):
        assert flux is not None
        self.Particles = flux
        self.logger.debug("Flux: %s", self.Particles)
        if forward_trck is not None:
            self.ForwardTracing = forward_trck
            return
        self.ForwardTracing = self.Particles.Mode.value
        self.logger.debug("Tracing: %s", "Inward" if self.ForwardTracing == 1 else "Outward")

    def __SetSave(self, Save):
        Nsave = Save if not isinstance(Save, list) else Save[0]
        self.Region.value.checkSave(self, Nsave)
        self.Npts = math.ceil(self.Num / Nsave) if Nsave != 0 else 1
        self.Nsave = Nsave
        self.logger.debug("Save every %s step of:", self.Nsave)
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

        for saves in self.Save.keys():
            self.logger.debug("\t%s: %s", saves, self.Save[saves])

    def __call__(self):
        Track = []
        self.logger.debug("Launching simulation...\n")
        file_nums = np.arange(self.Nfiles) if isinstance(self.Nfiles, int) else self.Nfiles
        for (idx, i) in enumerate(file_nums):
            if self.IsFirstRun:
                self.logger.info("File %d/%d started", idx + 1, len(file_nums))
                self.logger.debug("")
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
                self.logger.info("File %d/%d saved\n", idx + 1, len(file_nums))
                RetArr.clear()
            else:
                Track.append(RetArr)

        self.logger.info("Simulation completed!")
        if self.Output is None:
            return Track

    def CallOneFile(self):
        self.Particles.generate()
        RetArr = []

        SaveR = self.Save["Coordinates"]
        SaveV = self.Save["Velocities"]
        SaveE = self.Save["Efield"]
        SaveB = self.Save["Bfield"]
        SaveA = self.Save["Angles"]
        SaveP = self.Save["Path"]
        SaveD = self.Save["Density"]
        SaveC = self.Save["Clock"]
        SaveT = self.Save["Energy"]
        SavePA = self.Save["PitchAngles"]
        SaveLR = self.Save["LarmorRadii"]
        SaveGC = self.Save["GuidingCenter"]

        Gen = self.__gen
        GenMax = 1 if self.nuclear_interaction is None else self.nuclear_interaction.max_generations

        UseAdditionalEnergyLosses = self.Region.value.CalcAdditional()

        n_events = len(self.Particles)
        progress_step = self.Num // 10
        for self.index in range(n_events):
            self.logger.debug("Event %d/%d started", self.index + 1, n_events)
            TotTime, TotPathLen, TotPathDen = 0, 0, 0
            if self.Medium is not None and self.nuclear_interaction is not None:
                local_den, n_local, local_path_den = 0, 0, 0
                local_chem_comp = np.zeros(len(self.Medium.get_element_list()))
                local_path_den_vector = []
                local_coordinate = []
                local_velocity = []
            lon_total, lon_prev, full_revolutions = np.array([[0.]]), np.array([[0.]]), np.array([[0.]])
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
                self.logger.debug("Use Decay: %s", self.UseDecay)
                self.logger.debug("Decay rnd: %f", rnd_dec)

            if self.ForwardTracing == -1:
                self.logger.debug("Backtracing mode is ON")
                self.logger.debug("Redefinition of particle to antiparticle")
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

            self.logger.debug(
                "Particle: %s (M = %f [MeV/c2], Z = %d)",
                particle.Name, M, self.Particles[self.index].Z
            )
            self.logger.debug(
                "Energy: %f [MeV], Rigidity: %f [GV]",
                T, ConvertT2R(T, M, particle.A, particle.Z) / 1000 if particle.Z != 0 else np.inf
            )
            self.logger.debug("Coordinates: %s [m]", r)
            self.logger.debug("Velocity: %s", V_normalized)
            self.logger.debug("Beta: %s", V_norm / Constants.c)
            self.logger.debug("Beta * dt: %f [m]", V_norm * Step)

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

            self.logger.debug("Calculating:")

            PitchAngle = None
            LarmorRadius = None
            GuidingCenter = None

            for i in range(Num):
                if SavePA:
                    PitchAngle = functions.CalcPitchAngles(B, Vm)
                if SaveLR:
                    LarmorRadius = functions.CalcLarmorRadii(np.linalg.norm(B), T, PitchAngle, M, particle.Z)
                if SaveGC:
                    GuidingCenter = functions.CalcGuidingCenter(r, Vm, B, T, PitchAngle, M, particle.Z)

                if i % Nsave == 0 or i == Num - 1 or i_save == 0:
                    self._save_step(
                        r_old, V_norm, TotPathLen, TotPathDen, TotTime, Vm, r, T, E, B,
                        PitchAngle, LarmorRadius, GuidingCenter, Saves, self.SaveColumnLen,
                        self.SaveCode["Coordinates"], self.SaveCode["Velocities"], self.SaveCode["Efield"],
                        self.SaveCode["Bfield"], self.SaveCode["Angles"], self.SaveCode["Path"],
                        self.SaveCode["Density"], self.SaveCode["Clock"], self.SaveCode["Energy"],
                        self.SaveCode["PitchAngles"], self.SaveCode["LarmorRadii"], self.SaveCode["GuidingCenter"],
                        SaveR, SaveV, SaveE, SaveB, SaveA, SaveP, SaveD, SaveC, SaveT, SavePA, SaveLR, SaveGC
                    )
                    i_save += 1

                if self.UseAdaptiveStep:
                    Step = self._adaptive_step(Q, m, B, Vm, T, M, Step, self.N1, self.N2)
                    if i == 0:
                        self.Step = Step
                    q = Step * Q / 2 / m

                PathLen = V_norm * Step

                Vp, Yp, Ya = self.AlgoStep(T, M, q, Vm, r, B, E)

                if self.UseRadLosses[1]:
                    synch_record.add_iteration(T, B, Vm, Step)
                if self.UseRadLosses[0]:
                    Vm, T, new_photons, synch_record = RadLossStep.MakeRadLossStep(
                        Vp, Vm, Yp, Ya, M, Q, r, Step,
                        self.ForwardTracing, self.UseRadLosses[1:], particle, Gen, Constants, synch_record
                    )
                    prod_tracks.extend(new_photons)
                elif M > 0:
                    T = M * (Yp - 1)
                    Vm = Vp

                if UseAdditionalEnergyLosses:
                    Vm, T = self.Region.value.AdditionalEnergyLosses(r, Vm, T, M, Step, self.ForwardTracing,
                                                                     Constants.c)
                r_old = r
                V_norm, r, TotPathLen, TotTime = self._update(PathLen, Step, TotPathLen, TotTime, Vm, r)

                # Medium
                if self.Medium is not None:
                    self.Medium.calculate_model(*r)
                    Den = self.Medium.get_density()  # kg/m3
                    PathDen = (Den * 1e-3) * (PathLen * 1e2)  # g/cm2
                    TotPathDen += PathDen  # g/cm2
                    if self.nuclear_interaction is not None and Den > 0:
                        local_den += Den
                        local_chem_comp += self.Medium.get_element_abundance()
                        n_local += 1
                        local_path_den += PathDen
                        local_path_den_vector.append(local_path_den)
                        local_coordinate.append(r)
                        local_velocity.append(Vm)

                # Decay
                if self.UseDecay and not self.IsPrimDeath:
                    lifetime = tau * (T / M + 1) if M > 0 else np.inf
                    if rnd_dec > np.exp(-TotTime / lifetime):
                        self.__Decay(Gen, GenMax, T, TotTime, V_norm, Vm, particle, prod_tracks, r)
                        self.IsPrimDeath = True

                # Nuclear Interaction
                check_interaction = (
                    self.nuclear_interaction is not None
                    and local_path_den > self.nuclear_interaction.grammage_threshold
                    and not self.IsPrimDeath
                )
                if check_interaction:
                    # Construct Rotation Matrix & Save velocity before possible interaction
                    rotationMatrix = vecRotMat(np.array([0, 0, 1]), Vm / V_norm)
                    primary, secondary = self.nuclear_interaction.run_matter_layer(
                        pdg=particle.PDG,
                        energy=T,
                        mass=local_path_den,
                        density=(local_den * 1e-3) / n_local,
                        element_name=self.Medium.get_element_list(),
                        element_abundance=local_chem_comp / n_local
                    )
                    T = primary['KineticEnergy']
                    if T > 0 and T > 1:  # Cut particles with T < 1 MeV
                        # Only ionization losses
                        V_norm = Constants.c * np.sqrt(1 - (M / (T + M)) ** 2)
                        Vm = V_norm * rotationMatrix @ primary['MomentumDirection']
                        local_den, n_local, local_path_den = 0, 0, 0
                        local_chem_comp = np.zeros(len(self.Medium.get_element_list()))
                        local_path_den_vector.clear()
                        local_coordinate.clear()
                        local_velocity.clear()
                    else:
                        # Death due to ionization losses or nuclear interaction
                        self.IsPrimDeath = True
                        if secondary.size > 0 and Gen < GenMax:
                            self.logger.debug(
                                "Nuclear interaction %s: %d secondaries, total energy %f MeV",
                                primary["LastProcess"], secondary.size, np.sum(secondary["KineticEnergy"]),
                            )
                            self.logger.debug("%s", secondary)
                            # Coordinates of interaction point in XYZ
                            local_path_den_vector = np.array(local_path_den_vector)
                            path_den_cylinder = (np.linalg.norm(primary['Position']) * 1e2) * (local_den * 1e-3 / n_local)  # Path in cylinder [g/cm2]
                            r_interaction = np.array(local_coordinate)[np.argmax(local_path_den_vector > path_den_cylinder), :]
                            v_interaction = np.array(local_velocity)[np.argmax(local_path_den_vector > path_den_cylinder), :]
                            rotationMatrix = vecRotMat(np.array([0, 0, 1]), v_interaction / np.linalg.norm(v_interaction))
                            for p in secondary:
                                V_p = rotationMatrix @ p['MomentumDirection']
                                T_p = p['KineticEnergy']
                                PDGcode_p = p["PDGcode"]
                                # Parameters for recursive call of GT
                                params = self.ParamDict.copy()
                                params["Date"] += datetime.timedelta(seconds=TotTime)
                                params["Particles"] = Flux(
                                    Distribution=Distributions.UserInput(R0=r_interaction, V0=V_p),
                                    Spectrum=Spectrums.UserInput(energy=T_p),
                                    PDGcode=PDGcode_p
                                )
                                # if (PDGcode_p in self.nuclear_interaction.get("ExcludeParticleList", [])
                                #         or T_p < self.nuclear_interaction.get("Emin", 0)):
                                #     params["Num"] = 1
                                #     params["UseDecay"] = False
                                #     params["InteractNUC"] = None
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
                if self.Region == Regions.Magnetosphere:
                    if self.ParticleOriginIsOn or self.__brck_arr[self.__brck_index["MaxRev"]] != BreakDef[-1]:
                        a_, b_, _ = Functions.transformations.geo2mag_eccentric(GuidingCenter[0][0],
                                                                                GuidingCenter[0][1],
                                                                                GuidingCenter[0][2],
                                                                                1,
                                                                                self.Bfield.g,
                                                                                self.Bfield.h)
                        lon_total, lon_prev, full_revolutions = Additions.AddLon(lon_total, lon_prev, full_revolutions,
                                                                                 i, a_, b_)

                brck = self._check_break(r, r0, BCcenter, TotPathLen, TotTime, full_revolutions, BrckArr)
                brk = brck[1]
                if brck[0] or self.IsPrimDeath:
                    if SavePA:
                        PitchAngle = functions.CalcPitchAngles(B, Vm)
                    if SaveLR:
                        LarmorRadius = functions.CalcLarmorRadii(np.linalg.norm(B), T, PitchAngle, M, particle.Z)
                    if SaveGC:
                        GuidingCenter = functions.CalcGuidingCenter(r, Vm, B, T, PitchAngle, M, particle.Z)

                    if brk != -1:
                        self._save_step(
                            r_old, V_norm, TotPathLen, TotPathDen, TotTime, Vm, r, T, E, B, PitchAngle, LarmorRadius,
                            GuidingCenter, Saves, self.SaveColumnLen,
                            self.SaveCode["Coordinates"], self.SaveCode["Velocities"], self.SaveCode["Efield"],
                            self.SaveCode["Bfield"], self.SaveCode["Angles"], self.SaveCode["Path"],
                            self.SaveCode["Density"], self.SaveCode["Clock"], self.SaveCode["Energy"],
                            self.SaveCode["PitchAngles"], self.SaveCode["LarmorRadii"], self.SaveCode["GuidingCenter"],
                            SaveR, SaveV, SaveE, SaveB, SaveA, SaveP, SaveD, SaveC, SaveT, SavePA, SaveLR, SaveGC
                        )
                        i_save += 1
                    if self.IsPrimDeath:
                        brk = self.__brck_index["Death"]
                    self.logger.debug("### Break due to %s ###", self.__index_brck[brk])
                    break

                if i % progress_step == 0:
                    self.logger.debug("\tProgress: %d%%", int(i / self.Num * 100))

            self.logger.debug("\tProgress: 100%")
            if self.IsFirstRun:
                self.logger.info(
                    "Event %d/%d finished in %.3f seconds",
                    self.index + 1, n_events, timer() - st,
                )
            self.logger.debug("")

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
            if SavePA:
                track["PitchAngles"] = Saves[:, self.SaveCode["PitchAngles"]]
            if SaveLR:
                track["LarmorRadii"] = Saves[:, self.SaveCode["LarmorRadii"]]
            if SaveGC:
                track["GuidingCenter"] = Saves[:, self.SaveCode["GuidingCenter"]]

            RetArr.append({"Track": track,
                           "BC": {"WOut": brk},
                           "Particle": {"PDG": particle.PDG, "M": M, "Ze": particle.Z, "Gen": Gen,
                                        "R0": particle.coordinates, "V0": particle.velocities, "T0": particle.T},
                           "Child": prod_tracks})

            # TODO refactor
            if self.Region == Regions.Magnetosphere:
                # Particles in magnetosphere (Part 1)
                if self.TrackParamsIsOn:
                    self.logger.debug("Calculating additional parameters ...")
                    TrackParams_i = Additions.GetTrackParams(self, RetArr[self.index])
                    if self.__brck_arr[self.__brck_index["MaxRev"]] != BreakDef[-1]:
                        TrackParams_i["LonTotal"] = lon_total
                    RetArr[self.index]["Additions"] = TrackParams_i

                # Particles in magnetosphere (Part 2)
                if self.ParticleOriginIsOn and self.IsFirstRun:
                    self.logger.debug("Finding particle origin ...")
                    origin = Additions.FindParticleOrigin(self, RetArr[self.index])
                    RetArr[self.index]["Additions"]["ParticleOrigin"] = origin
                    self.logger.debug("Particle origin: %s", origin.name)

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
                params = self.ParamDict.copy()
                params["Particles"] = Flux(
                    Distribution=Distributions.UserInput(R0=r_p, V0=V_p),
                    Spectrum=Spectrums.UserInput(energy=T_p),
                    PDGcode=PDGcode_p
                )
                params["Date"] = params["Date"] + datetime.timedelta(seconds=TotTime)
                if PDGcode_p in [12, 14, 16, 18, -12, -14, -16, -18]:
                    params["Medium"] = None
                    params["InteractNUC"] = None
                new_process = self.__class__(**params)
                new_process.__gen = Gen + 1
                prod_tracks.append(new_process.CallOneFile()[0])

    @staticmethod
    @njit(fastmath=True)
    def _check_break(r, r0, center, TotPath, TotTime, full_revolutions, Brck):
        radi = np.linalg.norm(r - center)
        dst2path = np.linalg.norm(r - r0) / TotPath
        cond = np.concatenate((np.array([*np.abs(r), radi, dst2path]) < Brck[:5],
                               np.array([*np.abs(r), radi, TotPath, TotTime]) > Brck[5:-1],
                               np.array([full_revolutions[0][0]]) >= Brck[-1]))
        if np.any(cond):
            return True, np.where(cond)[0][0]
        return False, -1

    @staticmethod
    @njit(fastmath=True)
    def _update(PathLen, Step, TotPathLen, TotTime, Vm, r):
        V_norm = np.linalg.norm(Vm)
        r_new = r + Vm * Step
        TotTime += Step
        TotPathLen += PathLen
        return V_norm, r_new, TotPathLen, TotTime

    @staticmethod
    # @njit(fastmath=True)
    def _save_step(r_old, V_norm, TotPathLen, TotPathDen, TotTime, Vm, r, T, E, B,
                   PitchAngles, LarmorRadii, GuidingCenter,
                   Saves, ColLen,
                   RCode, VCode, ECode, BCode, ACode, PCode, DCode, CCode, TCode, PACode, LRCode, GCCode,
                   SaveR, SaveV, SaveE, SaveB, SaveA, SaveP, SaveD, SaveC, SaveT, SavePA, SaveLR, SaveGC):
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
        if SavePA:
            sv[PACode] = PitchAngles
        if SaveLR:
            sv[LRCode] = LarmorRadii
        if SaveGC:
            sv[GCCode] = GuidingCenter
        Saves.append(sv)

    @staticmethod
    @njit(fastmath=True)
    def _adaptive_step(q, m, B, V, T, M, dt, N1, N2):
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
