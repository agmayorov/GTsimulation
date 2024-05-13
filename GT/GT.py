import math

import matplotlib.pyplot as plt
import tqdm
from timeit import default_timer as timer
import numpy as np
import importlib

from datetime import datetime
from abc import ABC, abstractmethod

from numba import jit

from MagneticFields import Regions
from GT import Constants, Units


class GTSimulator(ABC):
    def __init__(self, Bfield=None, Efield=None, Region=Regions.Magnetosphere, Medium=None, Date=datetime(2008, 1, 1),
                 RadLosses=False, Particles="Monolines", ForwardTrck=None, Save: int | list = 1, Num: int = 1e6,
                 Step=1, Nfiles=1, Output=None, Verbose=False, BreakCondition: None | dict = None):
        self.Verbose = Verbose
        if self.Verbose:
            print("Creating simulator object...")

        self.Step = Step
        self.Num = int(Num)

        if self.Verbose:
            print(f"\tTime step: {self.Step}")
            print(f"\tNumber of steps: {self.Num}")

        self.Date = Date
        if self.Verbose:
            print(f"\tDate: {self.Date}")

        self.UseRadLosses = RadLosses
        if self.Verbose:
            print(f"\tRadiation Losses: {self.UseRadLosses}")

        self.Region = Region

        if self.Verbose:
            print(f"\tRegion: {self.Region.name}")

        self.ToMeters = 1
        self.Bfield = None
        self.Efield = None
        self.__SetEMFF(Bfield, Efield)

        self.Medium = None
        self.__SetMedium(Medium)

        self.Particles = None
        self.ForwardTracing = 1
        self.__SetFlux(Particles, ForwardTrck)

        self.Nfiles = 1 if Nfiles is None or Nfiles == 0 else Nfiles
        self.Output = Output
        self.Npts = 2
        self.Save = {"Clock": False, "Path": False, "Bfield": False, "Efield": False, "Energy": False, "Angles": False}
        self.__SetSave(Save)
        if self.Verbose:
            print(f"\tNumber of files: {self.Nfiles}")
            print(f"\tOutput file name: {self.Output}_file_num.npy")

        self.__brck_index = {"Xmin": 0, "Ymin": 1, "Zmin": 2, "Rmin": 3, "Dist2Path": 4, "Xmax": 5, "Ymax": 6,
                             "Zmax": 7, "Rmax": 8, "MaxPath": 9, "MaxTime": 10}
        self.__index_brck = {-1: "Loop", 0: "Xmin", 1: "Ymin", 2: "Zmin", 3:"Rmin", 4: "Dist2Path", 5: "Xmax",
                             6: "Ymax", 7: "Zmax", 8: "Rmax", 9: "MaxPath", 10: "MaxTime"}
        self.BrckArr = np.array([0, 0, 0, 0, 0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        self.__SetBrck(BreakCondition)

        self.index = 0
        if self.Verbose:
            print("Simulator created!\n")

    def __SetBrck(self, Brck):
        if Brck is not None:
            for key in Brck.keys():
                self.BrckArr[self.__brck_index[key]] = Brck[key]
        if self.Verbose:
            print("\tBreak Conditions: ")
            for key in self.__brck_index.keys():
                print(f"\t\t{key}: {self.BrckArr[self.__brck_index[key]]}")
        self.BrckArr[:-1] *= self.ToMeters

    def __SetMedium(self, medium):
        if self.Verbose:
            print("\tMedium: ", end='')
        if medium is not None:
            m = importlib.import_module("Medium.Magnetosphere.GTnrmlsise00")
            self.Medium = m.GTnrmlsise00(self.Date)
        else:
            if self.Verbose:
                print(None)

    def __SetFlux(self, flux, forward_trck):
        if self.Verbose:
            print("\tFlux:", end='')
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
            params = {'date': self.Date, "use_tesla": True, "use_meters": True,
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
        self.Npts = math.ceil(self.Num / Nsave)
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
        Wout = []
        if self.Verbose:
            print("Launching simulation...")
        for i in range(self.Nfiles):
            print(f"\tFile No {i + 1} of {self.Nfiles}")
            RetDict, wout = self.CallOneFile()

            if self.Output is not None:
                np.save(f"{self.Output}_{i}.npy", RetDict)
                if self.Verbose:
                    print("\tFile saved!")
            Track.append(RetDict)
            Wout.append(wout)
        if self.Verbose:
            print("Simulation completed!")
        return Track, Wout

    def CallOneFile(self):
        self.Particles.Generate()
        RetArr = []
        brk_arr = []
        SaveE = self.Save["Efield"]
        SaveB = self.Save["Bfield"]
        SaveA = self.Save["Angles"]
        SaveP = self.Save["Path"]
        SaveC = self.Save["Clock"]
        SaveT = self.Save["Energy"]
        for self.index in range(len(self.Particles)):
            if self.Verbose:
                print("\t\tStarting event...")
            TotTime, TotPathLen = 0, 0

            Saves = np.zeros((self.Npts + 1, 16))
            BrckArr = self.BrckArr

            Q = self.Particles[self.index].Z * Constants.e
            M = self.Particles[self.index].M
            E = self.Particles[self.index].E
            T = self.Particles[self.index].T

            r = np.array(self.Particles[self.index].coordinates)

            V_normalized = np.array(self.Particles[self.index].velocities)
            V_norm = Constants.c * np.sqrt(E ** 2 - M ** 2) / E
            Vm = V_norm * V_normalized

            if self.Verbose:
                print(f"\t\t\tParticle: {self.Particles[self.index].Name} (M = {M}, "
                      f"Z = {self.Particles[self.index].Z})")
                print(f"\t\t\tEnergy: {T} (beta = {V_norm / Constants.c})")
                print(f"\t\t\tCoordinates: {r / self.ToMeters}")
                print(f"\t\t\tVelocity: {V_normalized}")

            q = self.Step * Q / 2 / (M * Units.MeV2kg)
            brk = self.__index_brck[-1]
            Step = self.Step
            Num = self.Num
            Nsave = self.Nsave
            i_save = 0
            st = timer()
            if self.Verbose:
                print(f"\t\t\tCalculating: ", end=' ')
            for i in range(Num):
                PathLen = V_norm * Step

                Vp, Yp, Ya, B, E = self.AlgoStep(T, M, q, Vm, r)
                Vm, T = self.RadLossStep(Vp, Vm, Yp, Ya, M, Q)
                V_norm, r_new, TotPathLen, TotTime = self.Update(PathLen, Step, TotPathLen, TotTime, Vm, r)
                if i % Nsave == 0 or i == Num - 1 or i_save == 0:
                    self.SaveStep(r_new, V_norm, TotPathLen, TotTime, Vm, i_save, r, T, E, B, Saves,
                                  SaveE,
                                  SaveB,
                                  SaveA,
                                  SaveP,
                                  SaveC,
                                  SaveT)
                    i_save += 1
                r = r_new

                if self.Medium is not None:
                    rho = self.Medium.GetDensity(r_new)
                    a = 1

                if i % (self.Num // 100) == 0:
                    brck = self.CheckBreak(r, Saves[0, :3], TotPathLen, TotTime, BrckArr)
                    brk = self.__index_brck[brck[1]]
                    if brck[0]:
                        if self.Verbose:
                            print(f"Break due to {brk}", end=' ')
                        break

                if self.Verbose and (i / self.Num * 100) % 10 == 0:
                    print(f"{int(i / self.Num * 100)}%", end=' ')

            if self.Verbose:
                print("100%")
            print(f"\t\tEvent No {self.index + 1} of {len(self.Particles)} in {timer() - st} seconds")
            if self.Verbose:
                print()
            Saves = Saves[:i_save]
            Saves[:, :3] /= self.ToMeters

            ret = Saves[:, :6]

            if SaveE:
                ret = np.hstack((ret, Saves[:, 6:9]))
            if SaveB:
                ret = np.hstack((ret, Saves[:, 9:12]))
            if SaveA:
                ret = np.hstack((ret, Saves[:, 12][:, np.newaxis]))
            if SaveP:
                ret = np.hstack((ret, Saves[:, 13][:, np.newaxis]))
            if SaveC:
                ret = np.hstack((ret, Saves[:, 14][:, np.newaxis]))
            if SaveT:
                ret = np.hstack((ret, Saves[:, 15][:, np.newaxis]))
            RetArr.append(ret)
            brk_arr.append(brk)
        RetArr = np.array(RetArr)
        RetDict = {"Coordinates": RetArr[:, :, :3], "Velocities": RetArr[:, :, 3:6]}
        i = 6
        if SaveE:
            RetDict["Efield"] = RetArr[:, :, i:i + 3]
            i += 3
        if SaveB:
            RetDict["Bfield"] = RetArr[:, :, i:i + 3]
            i += 3
        if SaveA:
            RetDict["Angles"] = RetArr[:, :, i]
            i += 1
        if SaveP:
            RetDict["Path"] = RetArr[:, :, i]
            i += 1
        if SaveC:
            RetDict["Clock"] = RetArr[:, :, i]
            i += 1
        if SaveT:
            RetDict["Energy"] = RetArr[:, :, i]
        return RetDict, brk_arr

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def CheckBreak(r, r0, TotPath, TotTime, Brck):
        radi = np.linalg.norm(r)
        dst2path = np.linalg.norm(r - r0) / TotPath
        cond = np.concatenate((np.array([*np.abs(r), radi, dst2path]) < Brck[:5],
                               np.array([*np.abs(r), radi, TotPath, TotTime]) > Brck[5:]))
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
    def SaveStep(r_new, V_norm, TotPathLen, TotTime, Vm, i_save, r, T, E, B, Saves,
                 SaveE,
                 SaveB,
                 SaveA,
                 SaveP,
                 SaveC,
                 SaveT
                 ):
        Saves[i_save, :3] = r
        Saves[i_save, 3:6] = Vm / V_norm
        if SaveE:
            Saves[i_save, 6:9] = E
        if SaveB:
            Saves[i_save, 9:12] = B
        if SaveA:
            Saves[i_save, 12] = np.arctan2(np.linalg.norm(np.cross(r, r_new)), np.dot(r, r_new))
        if SaveP:
            Saves[i_save, 13] = TotPathLen
        if SaveC:
            Saves[i_save, 14] = TotTime
        if SaveT:
            Saves[i_save, 15] = T

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
