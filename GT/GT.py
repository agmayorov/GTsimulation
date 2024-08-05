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

from GeantInteraction import G4Decay
from MagneticFields.Magnetosphere.Functions import transformations
from Global import Constants, Units, Regions, Origins, BreakCode, BreakIndex, SaveCode, SaveDef, BreakDef, \
    BreakMetric, SaveMetric, vecRotMat
from Particle import ConvertT2R, GetAntiParticle
from .functions import GetLastPoints


class GTSimulator(ABC):
    def __init__(self, Bfield=None, Efield=None, Region=Regions.Magnetosphere, Medium=None, Date=datetime.datetime(2008, 1, 1),
                 RadLosses=False, Particles="Monolines", TrackParams=False, ParticleOrigin=False, IsFirstRun=True, Backtracing=False,
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

        self.Backtracing = Backtracing

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

    def refreshParams(self, Bfield=None, Efield=None, Region=Regions.Magnetosphere, Medium=None, Date=datetime.datetime(2008, 1, 1),
                 RadLosses=False, Particles="Monolines", TrackParams=False, ParticleOrigin=False, IsFirstRun=True, Backtracing=False,
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
        if self.ParticleOrigin:
            self.ParticleOriginIsOn = True
            self.TrackParams = True
            self.TrackParamsIsOn = True
        else:
            self.TrackParams = TrackParams
        if self.TrackParams:
            self.TrackParamsIsOn = True

        self.IsFirstRun = IsFirstRun
        self.Backtracing = Backtracing

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
        self.InteractNUC = UseInteractNUC
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
        self.BCcenter = center*self.ToMeters

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
        params = {"ToMeters": ToMeters, **({} if not isinstance(flux, list) else flux[1])}
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

            if self.TrackParams:
                self.GTmag = []

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
        GenMax = self.InteractNUC["GenMax"]

        for self.index in range(len(self.Particles)):
            if self.Verbose:
                print("\t\tStarting event...")
            TotTime, TotPathLen, TotPathDen = 0, 0, 0
            lon_total, lon_prev, full_revolutions = 0, 0, 0
            particle = self.Particles[self.index]
            Saves = np.zeros((self.Npts + 1, 17))
            BrckArr = self.__brck_arr
            BCcenter = self.BCcenter
            tau = self.UseDecay * particle.tau
            rnd_dec = 0
            IsPrimDecay = 0
            prod_tracks = []
            if tau:
                rnd_dec = np.random.rand()
                if self.Verbose:
                    print(f"\t\t\tUse Decay: {self.UseDecay}")
                    print(f"\t\t\tDecay rnd: {rnd_dec}")

            if self.Backtracing:
                GetAntiParticle(particle)
                particle.velocities = -particle.velocities

            Q = particle.Z * Constants.e
            M = particle.M
            E = particle.E
            T = particle.T

            r = np.array(particle.coordinates)

            V_normalized = np.array(particle.velocities)
            V_norm = Constants.c * np.sqrt(E ** 2 - M ** 2) / E
            Vm = V_norm * V_normalized

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

                # Decay
                if tau:
                    lifetime = tau * (T/M + 1)
                    if rnd_dec > np.exp(-TotTime/lifetime):
                        self.__Decay(Gen, GenMax, T, TotTime, V_norm, Vm, particle, prod_tracks, r)
                        IsPrimDecay = True

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
                    a_, b_, _ = transformations.geo2mag_eccentric(r[0], r[1], r[2], 1, self.ParamDict["Date"])
                    lon = np.arctan(b_ / a_)
                    lon_tolerance = np.deg2rad(60)

                    if i == 0:
                        lon_prev = lon

                    lon_diff = np.abs(lon - lon_prev)
                    if lon_diff > np.pi / 2:
                        lon_diff = np.pi - lon_diff

                    if lon_diff > lon_tolerance:
                        lon_total += lon_diff
                        lon_prev = lon

                    lon_total = np.abs(lon_total)

                    if lon_total > 2 * np.pi:
                        full_revolutions += 1
                        lon_total = np.remainder(lon_total, 2 * np.pi)

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
                    print("\t\t\tGet trajectory parameters (GTmag) ...", end=' ')
                self.__GetTrackParams(RetArr[self.index])
                if self.Verbose:
                    print("Done")

            # Particles in magnetosphere (Part 2)
            if self.ParticleOriginIsOn and self.IsFirstRun:
                if self.Verbose:
                    print("\t\t\tGet particle origin ...", end=' ')
                origin = self.__FindParticleOrigin(RetArr[self.index])
                self.GTmag[self.index]["ParicleOrigin"] = origin
                if self.Verbose:
                    print(origin.name)
                    print()
        return RetArr

    def __Decay(self, Gen, GenMax, T, TotTime, V_norm, Vm, particle, prod_tracks, r):
        if Gen < GenMax:
            product = G4Decay(particle.PDG, T / 1e3)
            rotationMatrix = vecRotMat(np.array([0, 0, 1]), Vm / V_norm)
            for p in range(len(product) - 1, -1, -1):
                prod = product[p]
                v_prod = rotationMatrix @ np.array(prod['v'])
                r_prod = r / self.ToMeters
                T_prod = prod['E'] * 1e3
                name_prod = prod["ParticleName"]

                params = self.ParamDict.copy()
                params["Particles"] = ["Monolines", {"Names": name_prod,
                                                     "T": T_prod,
                                                     "Center": r_prod,
                                                     "Radius": 0,
                                                     "V0": v_prod,
                                                     "Nevents": 1}]
                params["Date"] = params["Date"] + datetime.timedelta(seconds=TotTime)
                new_proc = self.__class__(**params)
                new_proc.__gen = Gen + 1
                prod_tracks.append(new_proc.CallOneFile())

    def __FieldLine(self, Rinp, sgn):
        bline = np.array([[0, 0, 0]])
        rline = np.array([Rinp])
        s = 0

        scale = 1e3 * 6370e3 / np.linalg.norm(Rinp)

        while np.linalg.norm(rline[s, :]) > Units.RE2m:
            B1, B2, B3 = self.Bfield.GetBfield(rline[s, 0], rline[s, 1], rline[s, 2])
            B = np.sqrt(B1 ** 2 + B2 ** 2 + B3 ** 2)
            d1 = sgn * B1 / B
            d2 = sgn * B2 / B
            d3 = sgn * B3 / B

            bline = np.append(bline, np.array([[B1, B2, B3]]), axis=0)
            rline = np.append(rline, rline[s, :] + Units.RE2m * np.array([[d1, d2, d3]]) / scale, axis=0)
            s += 1

        return rline[1:], bline[1:]

    def __GetEarthBfieldLine(self, rinp):
        urline, ubline = self.__FieldLine(rinp, 1)
        drline, dbline = self.__FieldLine(rinp, -1)

        rline = np.concatenate((urline.T, drline.T), axis=1).T
        bline = np.concatenate((ubline.T, dbline.T), axis=1).T

        return rline, bline

    def __GetTrackParams(self, RetArr_i):
        R = RetArr_i["Track"]["Coordinates"] * Units.RE2m
        H = RetArr_i["Track"]["Bfield"][1:, :]
        M = RetArr_i["Particle"]["M"] * Units.MeV2kg
        T0 = RetArr_i["Particle"]["T0"]
        Z = RetArr_i["Particle"]["Ze"]

        V = (R[1:, :] - R[:-1, :]) / self.Step
        Vn = np.linalg.norm(V, axis=1)
        R = R[1:, :]
        Hn = np.linalg.norm(H, axis=1)
        Y = 1 / np.sqrt((1 - Vn ** 2 / Constants.c ** 2))
        VdotH = np.sum(V * H, axis=1)

        # First invariant
        I1 = M * Y ** 2 * (Vn ** 2 - (VdotH / Hn) ** 2) / (2 * Hn)

        VndotHn = Vn * Hn

        # Pitch angles
        pitch = np.arccos(VdotH / VndotHn) / np.pi * 180

        # Mirror points
        a = pitch[1:] - 90
        a = np.where((pitch[:-1] - 90) * a < 0)[0]
        pitch_bound_tol = 0.4
        n = 0

        i = 0
        while i < a.size - 1:
            if np.max(np.abs(pitch[a[n]:a[n + 1]] - 90)) < pitch_bound_tol:
                a = np.delete(a, n + 1)
            else:
                n += 1
            i += 1

        num_mirror = None
        num_eq_pitch = None
        Hm = None
        Heq = None
        pitch_eq = None
        num_B0 = None

        if a.size > 0:
            num_mirror = np.zeros(a.size, dtype=int)
            num_eq_pitch = np.zeros(a.size - 1, dtype=int)
            num_B0 = np.zeros(a.size - 1, dtype=int)

            num_eq_1 = 0
            for i in range(a.size - 1):
                b = np.argmax(np.abs(pitch[a[i]:a[i + 1]] - 90))
                num_eq_2 = a[i] + b
                d = np.argmax(Hn[num_eq_1:num_eq_2 + 1])
                num_mirror[i] = num_eq_1 + d
                num_eq_1 = num_eq_2
                num_eq_pitch[i] = num_eq_2

                min_index = np.argmin(Hn[a[i]:a[i + 1] + 1])
                num_B0[i] = a[i] + min_index

            d = np.argmax(Hn[num_eq_1:])
            num_mirror[-1] = num_eq_1 + d
            if num_mirror.size == 0:
                num_mirror = None
            else:
                num_mirror = np.unique(num_mirror)
                Hm = Hn[num_mirror]

            if num_eq_pitch.size == 0:
                num_eq_pitch = None
                Heq = None
                pitch_eq = None
            else:
                Heq = Hn[num_eq_pitch]
                pitch_eq = pitch[num_eq_pitch]

            if num_B0.size == 0:
                num_B0 = None

        NumMirror = {"NumMirr": num_mirror, "NumEqPitch": num_eq_pitch, "NumBo": num_B0, "Hmirr": Hm, "Heq": Heq}

        # Second invariant
        I2 = None
        if num_mirror is not None:
            I2 = np.zeros(num_mirror.size - 1)
            I2_tol = 0.2

            for i in range(num_mirror.size - 1):
                HmTmp = max(Hm[i], Hm[i + 1])
                H_coil = Hn[num_mirror[i]:num_mirror[i + 1]]
                b = np.array([R[:, 0][num_mirror[i] + 1:num_mirror[i + 1] + 1],
                              R[:, 1][num_mirror[i] + 1:num_mirror[i + 1] + 1],
                              R[:, 2][num_mirror[i] + 1:num_mirror[i + 1] + 1]])
                S = b - np.array([R[:, 0][num_mirror[i]:num_mirror[i + 1]],
                                  R[:, 1][num_mirror[i]:num_mirror[i + 1]],
                                  R[:, 2][num_mirror[i]:num_mirror[i + 1]]])

                I2[i] = np.sum(
                    np.sqrt(1 - H_coil / HmTmp) * np.abs((S[0, :] * H[:, 0][num_mirror[i]:num_mirror[i + 1]] +
                                                          S[1, :] * H[:, 1][num_mirror[i]:num_mirror[i + 1]] +
                                                          S[2, :] * H[:, 2][num_mirror[i]:num_mirror[i + 1]]) /
                                                         H_coil))

            if I2.size == 0:
                I2 = None
            else:
                I2 = I2[I2 > I2_tol * np.max(I2)]

        if I2 is not None and I2.size == 0:
            I2 = None

        # L-shell and Guiding Centre
        RE = Units.RE2m
        k0 = 31100 * 1e-5  # Gauss * RE^3
        k0 *= 1e-4 * RE ** 3  # Tesla * m^3

        a_new = np.array([
            [3.0062102e-1, 6.2337691e-1, 6.228644e-1, 6.222355e-1, 2.0007187, -3.0460681],
            [3.33338e-1, 4.3432642e-1, 4.3352788e-1, 4.3510529e-1, -1.8461796e-1, 1],
            [0, 1.5017245e-2, 1.4492441e-2, 1.2817956e-2, 1.2038224e-1, 0],
            [0, 1.3714667e-3, 1.1784234e-3, 2.1680398e-3, -6.7310339e-3, 0],
            [0, 8.2711096e-5, 3.8379917e-5, -3.2077032e-4, 2.170224e-4, 0],
            [0, 3.2916354e-6, -3.3408822e-6, 7.9451313e-5, -3.8049276e-6, 0],
            [0, 8.1048663e-8, -5.3977642e-7, -1.2531932e-5, 2.8212095e-8, 0],
            [0, 1.0066362e-9, -2.1997983e-8, 9.9766148e-7, 0, 0],
            [0, 8.3232531e-13, 2.3028767e-9, -3.958306e-8, 0, 0],
            [0, -8.1537735e-14, 2.6047023e-10, 6.3271665e-10, 0, 0]
        ])

        L_shell = None
        GuidingCentre = {}
        LR = None
        LRNit = None
        parReq = None
        parBeq = None
        parBBo = None
        Req = None
        Beq = None
        BBo = None
        Rline = None
        Bline = None

        if self.IsFirstRun:
            if I2 is not None:
                I = np.mean(I2)
                Bm = np.mean(Hm)
                X = np.log(I ** 3 * Bm / k0)

                an = (a_new[:, 0] * (X < -22) + a_new[:, 1] * (-22 < X < 3) + a_new[:, 2] * (-3 < X < 3) + a_new[:,
                                                                                                           3] * (
                              3 < X < 11.7)
                      + a_new[:, 4] * (11.7 < X < 23)) + a_new[:, 5] * (X > 23)

                Y = np.sum(an * X ** np.arange(10))
                L_shell = (k0 / RE ** 3 / Bm * (1 + np.exp(Y))) ** (1 / 3)
                M /= Units.MeV2kg

                # Magnetic field line of Guiding Centre
                gamma = (T0 + M) / M
                omega = np.abs(Z) * Constants.e * Hn[0] / (gamma * M * Units.MeV2kg)

                # Larmor Radius
                LR = np.sin(np.deg2rad(pitch[0])) * np.sqrt(1 - 1 / gamma ** 2) * Constants.c / omega
                LRNit = 2 * np.pi * LR / (Vn[0] * self.Step)

                Nit = min(LRNit + 1, len(R))
                Nit = np.floor(np.arange(0, Nit, Nit / 3 - 1)).astype(int)
                Rmin = np.zeros((Nit.size, 3))

                for i in range(Nit.size):
                    Rline, Bline = self.__GetEarthBfieldLine(R[Nit[i], :])
                    Bn = np.linalg.norm(Bline, axis=1)
                    e = np.argmin(Bn)
                    Rmin[i, :] = Rline[e, :]
                    if i == 0:
                        parReq = Rline[e, :]
                        parBeq = Bline[e, :]
                        parBBo = np.linalg.norm(H[0, :]) / np.linalg.norm(parBeq)

                Rline, Bline = self.__GetEarthBfieldLine(np.mean(Rmin, axis=0))
                Bn = np.linalg.norm(Bline, axis=1)
                e = np.argmin(Bn)
                Req = Rline[e, :]
                Beq = Bline[e, :]
                BBo = np.linalg.norm(H[0, :]) / np.linalg.norm(Beq)

                parReqNew = transformations.geo2mag_eccentric(parReq[0], parReq[1], parReq[2], 1,
                                                              self.ParamDict["Date"])
                GuidingCentre["parL"] = np.linalg.norm(parReqNew) / Units.RE2m

                ReqNew = transformations.geo2mag_eccentric(Req[0], Req[1], Req[2], 1, self.ParamDict["Date"])
                GuidingCentre["L"] = np.linalg.norm(ReqNew) / Units.RE2m

                parReq = parReq / Units.RE2m
                Req = Req / Units.RE2m
                Rline = Rline / Units.RE2m

        GuidingCentre = GuidingCentre | {"LR": LR, "LRNit": LRNit, "parReq": parReq, "parBeq": parBeq, "parBBo": parBBo,
                                         "Req": Req, "Beq": Beq, "BBo": BBo, "Rline": Rline, "Bline": Bline}

        TrackParams_i = {"Invariants": {"I1": I1, "I2": I2},
                         "PitchAngles": {"Pitch": pitch, "PitchEq": pitch_eq},
                         "MirrorPoints": NumMirror,
                         "GuidingCentre": GuidingCentre}

        self.GTmag.append(TrackParams_i)

    def __GetParticleOrigin(self, TrackParams_i):
        InitEndFlag = TrackParams_i["InitEndFlag"]
        isFullRevolution = TrackParams_i["isFullRevolution"]
        I1, I2 = TrackParams_i["I1"], TrackParams_i["I2"]
        num_mirror = TrackParams_i["Nm"]
        first_invariant_disp = TrackParams_i["I1_disp"]
        second_invariant_disp = TrackParams_i["I2_disp"]

        first_invariant_disp_tol = 0.3
        first_invariant_disp_tol_2 = 0.7
        second_invariant_disp_tol = 0.3
        full_revolution_tol = 5

        if InitEndFlag[0] == 2:
            origin = Origins.Galactic
            return origin
        if InitEndFlag[0] == 1 and InitEndFlag[1] == 2:
            origin = Origins.Albedo
            return origin
        if InitEndFlag[0] == 3 and InitEndFlag[1] == 2:
            origin = Origins.Unknown
            return origin

        if not isFullRevolution and (InitEndFlag[0] == 3 and InitEndFlag[1] == 3):
            origin = Origins.Unknown
            return origin

        if not isFullRevolution:
            if InitEndFlag[0] == 1 and InitEndFlag[1] == 1:
                if num_mirror is None:
                    origin = Origins.Albedo
                if 1 <= num_mirror.size <= 2:
                    if first_invariant_disp < first_invariant_disp_tol_2:
                        origin = Origins.Presipitated
                    else:
                        origin = Origins.Albedo
                if num_mirror.size > 2:
                    if first_invariant_disp < first_invariant_disp_tol_2:
                        origin = Origins.QuasiTrapped
                    else:
                        origin = Origins.Albedo
                return origin
            if num_mirror is not None and num_mirror.size <= 2:
                origin = Origins.Unknown
            elif num_mirror is not None and num_mirror.size > 2:
                if first_invariant_disp < first_invariant_disp_tol_2:
                    origin = Origins.QuasiTrapped
                else:
                    origin = Origins.Albedo
            return origin
        if isFullRevolution:
            if InitEndFlag[0] == 1 or InitEndFlag[1] == 1:
                if first_invariant_disp is not None:
                    if first_invariant_disp < first_invariant_disp_tol_2:
                        origin = Origins.Albedo
                    else:
                        origin = Origins.QuasiTrapped
                else:
                    origin = Origins.QuasiTrapped
                return origin
            if first_invariant_disp is not None and second_invariant_disp is not None:
                if (first_invariant_disp < first_invariant_disp_tol) and (
                        second_invariant_disp < second_invariant_disp_tol) or (isFullRevolution >= full_revolution_tol):
                    origin = Origins.Trapped
                else:
                    origin = Origins.Unknown
            else:
                origin = Origins.Unknown
        return origin

    def __GetBCparams(self, RetArr_i):
        if RetArr_i["BC"]["Status"] == "DefaultBC_Rmin":
            u = 1
        elif RetArr_i["BC"]["Status"] == "DefaultBC_Rmax":
            u = 2
        elif RetArr_i["BC"]["Status"] in ["UserBCfunction", "Done"]:
            u = 3
        else:
            u = 0

        lon_u = RetArr_i["BC"]["lon_total"]
        return u, lon_u

    def __AddTrajectory(self, f, b, lonTotal, lon, GTmag_i, Nm, I1, I2, s):
        InitEndFlag = np.array([f, b])
        lonTotal += lon
        isFullRevolution = 0 + (lonTotal > 2 * np.pi)

        if GTmag_i['Invariants']['I1'] is not None:
            if s == 1:
                I1 = np.concatenate((I1, GTmag_i['Invariants']['I1']))
            else:
                I1 = np.concatenate((np.flip(GTmag_i['Invariants']['I1'][:-1]), I1))

        d = (np.array([np.mean(I1)] * I1.size) - I1) ** 2
        I1_disp = np.sqrt(np.mean(d)) / np.mean(I1)

        # Mirror points
        if GTmag_i['MirrorPoints']['NumBo'] is not None:
            Nm = np.concatenate((Nm, GTmag_i['MirrorPoints']['NumBo']))

        # I2
        if GTmag_i['Invariants']['I2'] is not None:
            if s == 1:
                if GTmag_i['Invariants']['I2'] is not None:
                    I2 = np.concatenate((I2, GTmag_i['Invariants']['I2']))
            else:
                I2 = np.concatenate((np.flip(GTmag_i['Invariants']['I2'][:-1]), I2))

        if I2.size == 0:
            I2_disp = None
        else:
            d = (np.array([np.mean(I2)] * I2.size) - I2) ** 2
            I2_disp = np.sqrt(np.mean(d)) / np.mean(I2)

        addTrackDict = {"InitEndFlag": InitEndFlag, "lonTotal": lonTotal, "isFullRevolution": isFullRevolution,
                        "Nm": Nm, "I1": I1, "I1_disp": I1_disp, "I2": I2, "I2_disp": I2_disp}

        return addTrackDict

    def __FindParticleOrigin(self, RetArr_i):
        # Forward trajectory
        f, lon_f = self.__GetBCparams(RetArr_i)
        addTrackParams = self.__AddTrajectory(f, 0, 0, lon_f, self.GTmag[self.index], np.array([0]), np.array([]),
                                              np.array([]), 1)
        lon_total = addTrackParams["lonTotal"]
        Nm = addTrackParams["Nm"]
        I1 = addTrackParams["I1"]
        I2 = addTrackParams["I2"]

        Rf, Vf = GetLastPoints(RetArr_i, 1)

        # Backward trajectory
        Date = self.ParamDict["Date"]
        Region = self.ParamDict["Region"]
        Bfield = self.ParamDict["Bfield"]
        Particles = self.ParamDict["Particles"]
        Num = self.ParamDict["Num"]
        Step = self.ParamDict["Step"]
        Save = [1, {"Energy": True, "Bfield": True}]
        if isinstance(Particles, list):
            Particles[1]["Nevents"] = 1
        BreakCondition = self.ParamDict["BreakCondition"]
        if BreakCondition is None:
            BreakCondition = {"MaxRev": 5}
        else:
            BreakCondition["MaxRev"] = 5
        bGTsim = copy.deepcopy(self)
        bGTsim.refreshParams(Date=Date, Region=Region, Bfield=Bfield, Particles=Particles, Num=Num, Step=Step, Save=Save,
                             BreakCondition=BreakCondition, TrackParams=True, ParticleOrigin=False, IsFirstRun=False,
                             Backtracing=True)

        bGTtrack = bGTsim()
        b, lon_b = self.__GetBCparams(bGTtrack[0][0])

        addTrackParams = self.__AddTrajectory(f, b, lon_total, lon_b, bGTsim.GTmag[0], Nm, I1, I2, -1)

        Rb, Vb = GetLastPoints(bGTtrack[0][0], -1)

        # Determine the origin of the particle
        origin = self.__GetParticleOrigin(addTrackParams)

        # Repeat procedure
        while origin == Origins.Unknown:
            # Trace extension
            if f == 3:
                s = 1
                if not isinstance(Particles, list):
                    Particles = [Particles, {"Center": Rf, "V0": Vf}]
                else:
                    Particles[1]["Center"] = Rf
                    Particles[1]["V0"] = Vf
                fGTsim = copy.deepcopy(self)
                fGTsim.refreshParams(Date=Date, Region=Region, Bfield=Bfield, Particles=Particles, Num=Num, Step=Step,
                                     Save=Save, BreakCondition=BreakCondition, TrackParams=True, IsFirstRun=False)
                fGTtrack = fGTsim()

                Rf, Vf = GetLastPoints(fGTtrack[0][0], 1)
                f, lon = self.__GetBCparams(fGTtrack[0][0])
            else:
                if b == 3:
                    s = -1
                    if not isinstance(Particles, list):
                        Particles = [Particles, {"Center": Rb, "V0": Vb}]
                    else:
                        Particles[1]["Center"] = Rb
                        Particles[1]["V0"] = Vb

                    bGTsim.RefreshParams(Date=Date, Region=Region, Bfield=Bfield, Particles=Particles, Num=Num,
                                         Save=Save,
                                         Step=Step, BreakCondition=BreakCondition, TrackParams=True,
                                         ParticleOrigin=False,
                                         IsFirstRun=False,
                                         Backtracing=True)
                    bGTtrack = bGTsim()

                    Rb, Vb = GetLastPoints(bGTtrack[0][0], 1)
                    b, lon = self.__GetBCparams(bGTtrack[0][0])
                else:
                    break

            addTrackParams = self.__AddTrajectory(f, b, lon_total, lon, bGTsim.GTmag[0], Nm, I1, I2, -1)

            origin = self.__GetParticleOrigin(addTrackParams)

        return origin

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
