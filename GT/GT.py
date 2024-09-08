import os
import math

import matplotlib.pyplot as plt
import tqdm
from timeit import default_timer as timer
import numpy as np
import importlib
import datetime
import copy

from abc import ABC, abstractmethod

from numba import jit

from Interaction import G4Interaction, G4Decay
from MagneticFields.Magnetosphere import Functions, Additions
from Global import Constants, Units, Regions, BreakCode, BreakIndex, SaveCode, SaveDef, BreakDef, \
    BreakMetric, SaveMetric, vecRotMat
from Particle import ConvertT2R, GetAntiParticle


class GTSimulator(ABC):
    def __init__(self, Bfield=None, Efield=None, Region=Regions.Magnetosphere, Medium=None, Date=datetime.datetime(2008, 1, 1),
                 RadLosses=False, Particles="Monolines", TrackParams=False, ParticleOrigin=False, IsFirstRun=True,
                 ForwardTrck=None, Save: int | list = 1, Num: int = 1e6,
                 Step=1, Nfiles=1, Output=None, Verbose=False, BreakCondition: None | dict = None,
                 BCcenter=np.array([0, 0, 0]), UseDecay=False, InteractNUC: None | dict = None):

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

        self.UseRadLosses = RadLosses
        if self.Verbose:
            print(f"\tRadiation Losses: {self.UseRadLosses}")
            print()

        self.ParticleOrigin = ParticleOrigin
        self.ParticleOriginIsOn = False
        self.TrackParamsIsOn = False
        if self.ParticleOrigin:
            self.ParticleOriginIsOn = True
            self.TrackParams = True
            self.TrackParamsIsOn = True
        else:
            self.TrackParams = TrackParams
        if self.TrackParams:
            self.TrackParamsIsOn = True
        self.IsFirstRun = IsFirstRun

        self.Region = Region

        if self.Verbose:
            print(f"\tRegion: {self.Region.name}")
            print()

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
                 RadLosses=False, Particles="Monolines", TrackParams=False, ParticleOrigin=False, IsFirstRun=True,
                 ForwardTrck=None, Save: int | list = 1, Num: int = 1e6,
                 Step=1, Nfiles=1, Output=None, Verbose=False, BreakCondition: None | dict = None,
                 BCcenter=np.array([0, 0, 0]), UseDecay=False, InteractNUC: None | dict = None):

        self.__names = self.__init__.__code__.co_varnames[1:]
        self.__vals = []
        for self.__v in self.__names:
            self.__vals.append(eval(self.__v))
        self.ParamDict = dict(zip(self.__names, self.__vals))

        del self.__names, self.__vals

        self.Verbose = Verbose
        # if self.Verbose:
        #     print("Creating simulator object...")

        self.Date = Date
        # if self.Verbose:
        #     print(f"\tDate: {self.Date}")
        #     print()

        self.Step = Step
        self.Num = int(Num)

        # if self.Verbose:
        #     print(f"\tTime step: {self.Step}")
        #     print(f"\tNumber of steps: {self.Num}")
        #     print()

        self.UseRadLosses = RadLosses
        # if self.Verbose:
        #     print(f"\tRadiation Losses: {self.UseRadLosses}")
        #     print()

        self.ParticleOrigin = ParticleOrigin
        self.ParticleOriginIsOn = False
        self.TrackParamsIsOn = False
        if self.ParticleOrigin:
            self.ParticleOriginIsOn = True
            self.TrackParams = True
            self.TrackParamsIsOn = True
        else:
            self.TrackParams = TrackParams
        if self.TrackParams:
            self.TrackParamsIsOn = True
        self.IsFirstRun = IsFirstRun

        self.Region = Region

        # if self.Verbose:
        #     print(f"\tRegion: {self.Region.name}")
        #     print()

        self.ToMeters = 1
        self.Bfield = None
        self.Efield = None
        self.__SetEMFF(Bfield, Efield)
        # if self.Verbose:
        #     print()

        self.Medium = None
        self.__SetMedium(Medium)
        # if self.Verbose:
        #     print()

        self.UseDecay = False
        self.InteractNUC = None
        self.__gen = 1
        self.__SetNuclearInteractions(UseDecay, InteractNUC)
        # if self.Verbose:
        #     print()

        self.Particles = None
        self.ForwardTracing = 1
        self.__SetFlux(Particles, ForwardTrck)
        # if self.Verbose:
        #     print()

        self.Nfiles = 1 if Nfiles is None or Nfiles == 0 else Nfiles
        self.Output = Output
        self.Npts = 2
        self.Save = SaveDef.copy()
        # if self.Verbose:
        #     print(f"\tNumber of files: {self.Nfiles}")
        #     print(f"\tOutput file name: {self.Output}_file_num.npy")
        self.__SetSave(Save)
        # if self.Verbose:
        #     print()

        self.__brck_index = BreakCode.copy()
        self.__brck_index.pop("Loop")
        self.__index_brck = BreakIndex.copy()
        self.__brck_arr = BreakDef.copy()
        self.__SetBrck(BreakCondition, BCcenter)

        self.index = 0
        # if self.Verbose:
        #     print("Simulator created!\n")

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
        class_name = flux if not isinstance(flux, list) else flux[0]
        ToMeters = self.ToMeters
        if isinstance(flux, list):
            transform = flux[1].pop("Transform", None)
            if transform is not None:
                center = flux[1].get("Center", None)
                assert center is None
                center = self.Region.value.transform(*transform[1], transform[0], ToMeters)
                flux[1]["Center"] = center
        params = {"ToMeters": ToMeters, **({} if not isinstance(flux, list) else flux[1])}
        self.Region.value.transform()
        if hasattr(m, class_name):
            flux = getattr(m, class_name)
            self.Particles = flux(**params)
        else:
            raise Exception("No spectrum")

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

    def __SetSave(self, Save):
        Nsave = Save if not isinstance(Save, list) else Save[0]
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

        for self.index in range(len(self.Particles)):
            if self.Verbose:
                print("\t\tStarting event...")
            TotTime, TotPathLen, TotPathDen = 0, 0, 0
            if self.Medium is not None and self.InteractNUC is not None:
                LocalDen, LocalChemComp, nLocal, LocalPathDen = 0, np.zeros(len(self.Medium.chemical_element_list)), 0, 0
            lon_total, lon_prev, full_revolutions = np.array([[0.]]), np.array([[0.]]), 0
            particle = self.Particles[self.index]
            Saves = np.zeros((self.Npts + 1, 17))
            BrckArr = self.__brck_arr
            BCcenter = self.BCcenter
            tau = self.UseDecay * particle.tau
            rnd_dec = 0
            IsPrimDecay, IsPrimInteraction = False, False
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

            r = np.array(particle.coordinates)

            V_normalized = np.array(particle.velocities) # unit vector of velosity (beta vector)
            V_norm = Constants.c * np.sqrt(E ** 2 - M ** 2) / E # scalar speed [m/s]
            Vm = V_norm * V_normalized # vector of velocity [m/s]

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

            q = self.Step * Q / 2 / (M * Units.MeV2kg) if M != 0 else 0
            brk = BreakCode["Loop"]
            Step = self.Step
            Num = self.Num
            Nsave = self.Nsave if self.Nsave != 0 else Num + 1
            i_save = 0
            st = timer()
            if self.Verbose:
                print(f"\t\t\tCalculating: ", end=' ')
            for i in range(Num):
                PathLen = V_norm * Step

                Vp, Yp, Ya, B, E = self.AlgoStep(T, M, q, Vm, r)
                Vm, T = self.RadLossStep(Vp, Vm, Yp, Ya, M, Q)

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

                # Decay
                if tau:
                    lifetime = tau * (T/M + 1)
                    if rnd_dec > np.exp(-TotTime/lifetime):
                        self.__Decay(Gen, GenMax, T, TotTime, V_norm, Vm, particle, prod_tracks, r)
                        IsPrimDecay = True

                # Nuclear Interaction
                if self.InteractNUC is not None and LocalPathDen > self.IntPathDen and not IsPrimDecay:
                    # Construct Rotation Matrix & Save velosity before possible interaction
                    rotationMatrix = vecRotMat(np.array([0, 0, 1]), Vm / V_norm)
                    primary, secondary = G4Interaction(particle.PDG, T, LocalPathDen, LocalDen / nLocal, LocalChemComp / nLocal)
                    if primary['KineticEnergy'] > 0:
                        # Only ionization losses
                        T = primary['KineticEnergy']
                        Vm = rotationMatrix @ primary['MomentumDirection']
                        # Vm *= Constants.c * np.sqrt(E ** 2 - M ** 2) / E # scalar speed [m/s]
                        # V_norm = np.linalg.norm(Vm)
                        # Vcorr = c * sqrt(1 - (M / (T + M))^2) / V_norm
                        # Vm *= Vcorr
                        LocalDen, LocalChemComp, nLocal, LocalPathDen = 0, np.zeros(len(self.Medium.chemical_element_list)), 0, 0
                    else:
                        # Death due to ionization losses or nuclear interaction
                        # print('CHILDREN:', secondary.size)
                        # some code here
                        IsPrimInteraction = True
                        pass

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

                # Full revolution
                if self.ParticleOriginIsOn or self.__brck_arr[self.__brck_index["MaxRev"]] != BreakDef[-1]:
                    a_, b_, _ = Functions.transformations.geo2mag_eccentric(r[0], r[1], r[2], 1, self.ParamDict["Date"])
                    lon_total, lon_prev, full_revolutions = Additions.AddLon(lon_total, lon_prev, full_revolutions, i, a_, b_)

                # if i % (self.Num // 100) == 0:
                brck = self.CheckBreak(r, Saves[0, :3], BCcenter, TotPathLen, TotTime, full_revolutions, BrckArr)
                brk = brck[1]
                if brck[0] or IsPrimDecay:
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
                    if IsPrimDecay:
                        brk = self.__brck_index["Decay"]
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
                           "BC": {"WOut": brk, "lon_total": lon_total, "Status": status},
                           "Particle": {"PDG": particle.PDG, "M": M, "Ze": particle.Z, "T0": particle.T, "Gen": Gen},
                           "Child": prod_tracks})

            # Particles in magnetosphere (Part 1)
            if self.TrackParamsIsOn:
                if self.Verbose:
                    print("\t\t\tGet trajectory parameters ...", end=' ')
                TrackParams_i = Additions.GetTrackParams(self, RetArr[self.index])
                RetArr[self.index]["Additions"] = TrackParams_i
                if self.Verbose:
                    print("Done")

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
                params["Particles"] = ["Monolines", {"Names": name_p,
                                                     "T": T_p,
                                                     "Center": r_p,
                                                     "Radius": 0,
                                                     "V0": V_p,
                                                     "Nevents": 1}]
                params["Date"] = params["Date"] + datetime.timedelta(seconds=TotTime)
                new_process = self.__class__(**params)
                new_process.__gen = Gen + 1
                prod_tracks.append(new_process.CallOneFile())

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

    def RadLossStep(self, Vp, Vm, Yp, Ya, M, Q):
        if not self.UseRadLosses:
            T = M * (Yp - 1)
            return Vp, T

        acc = (Vp - Vm) / self.Step
        Vn = np.linalg.norm(Vp + Vm)
        Vinter = (Vp + Vm) / Vn

        acc_par = np.dot(acc, Vinter)
        acc_per = np.sqrt(np.linalg.norm(acc) ** 2 - acc_par ** 2)

        dE = self.Step * ((2 / (3 * 4 * np.pi * 8.854187e-12) * Q ** 2 * Ya ** 4 / Constants.c ** 3) *
                          (acc_per ** 2 + acc_par ** 2 * Ya ** 2) / Constants.e / 1e6)

        T = M * (Yp - 1) - self.ForwardTracing * np.abs(dE)

        V = Constants.c * np.sqrt((T + M) ** 2 - M ** 2) / (T + M)
        Vn = np.linalg.norm(Vp)

        Vm = V * Vp / Vn

        return Vm, T

    @abstractmethod
    def AlgoStep(self, T, M, q, Vm, r):
        pass

# if self.Save is None:
#     ax = plt.figure().add_subplot(projection='3d')
#     ax.plot(0, 0, 0, '*')
#     print(Trajectory[-1, 0], Trajectory[-1, 1], Trajectory[-1, 2])
#     print(TotPathLen / 1000)
#     ax.plot(Trajectory[0, 0], Trajectory[0, 1], Trajectory[0, 2], 'o')
#     ax.plot(Trajectory[-1, 0], Trajectory[-1, 1], Trajectory[-1, 2], 'o')
#     ax.plot(Trajectory[:, 0], Trajectory[:, 1], Trajectory[:, 2])
#     ax.set_xlim([-1, 1.5])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
#     plt.show()
