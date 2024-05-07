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
                 Step=1, Nfiles=1, Output=None):
        self.Step = Step
        self.Num = int(Num)

        self.Date = Date
        self.UseRadLosses = RadLosses

        self.Region = Region
        self.Bfield = None
        self.Efield = None
        self.__SetEMFF(Bfield, Efield)

        self.Medium = None
        self.__SetMedium(Medium)

        self.Particles = None
        self.ForwardTracing = 1
        self.__SetFlux(Particles, ForwardTrck)

        self.Nfiles = Nfiles if Nfiles is not None or Nfiles != 0 else 1
        self.Output = Output
        self.Npts = 2
        self.Save = {"Clock": False, "Path": False, "Bfield": False, "Efield": False, "Energy": False, "Angles": False}
        self.__SetSave(Save)

        self.index = 0

    def __SetMedium(self, medium):
        pass

    def __SetFlux(self, flux, forward_trck):
        module_name = f"Particle.Generators"
        m = importlib.import_module(module_name)
        class_name = flux if not isinstance(flux, list) else flux[0]
        ToMeters = 1 if self.Bfield is None else self.Bfield.ToMeters
        params = {"ToMeters": ToMeters, **({} if not isinstance(flux, list) else flux[1])}
        if hasattr(m, class_name):
            flux = getattr(m, class_name)
            self.Particles = flux(**params)
        else:
            raise Exception("No spectrum")

        if forward_trck is not None:
            self.ForwardTracing = forward_trck
            return

        self.ForwardTracing = self.Particles.Mode.value

    def __SetEMFF(self, Bfield=None, Efield=None):
        if Efield is not None:
            pass
        if Bfield is not None:
            module_name = f"MagneticFields.{self.Region.name}"
            m = importlib.import_module(module_name)
            class_name = Bfield if not isinstance(Bfield, list) else Bfield[0]
            params = {'date': self.Date, "use_tesla": True, "use_meters": True,
                      **({} if not isinstance(Bfield, list) else Bfield[1])}
            if hasattr(m, class_name):
                B = getattr(m, class_name)
                self.Bfield = B(**params)
            else:
                raise Exception("No such field")

    def __SetSave(self, Save):
        Nsave = Save if not isinstance(Save, list) else Save[0]
        self.Npts = math.ceil(self.Num / Nsave)
        self.Nsave = Nsave
        if isinstance(Save, list):
            for saves in Save[1].keys():
                self.Save[saves] = Save[1][saves]

    def __call__(self):
        Track = []
        for i in range(self.Nfiles):
            print(f"File No {i+1} of {self.Nfiles}")
            RetArr = self.CallOneFile()

            if self.Output is not None:
                np.save(f"{self.Output}_{i}.npy", RetArr)
            Track.append(RetArr)
        return Track

    def CallOneFile(self):
        self.Particles.Generate()
        RetArr = []
        for self.index in range(len(self.Particles)):
            TotTime, TotPathLen = 0, 0

            Saves = np.zeros((self.Npts + 1, 16))

            SaveE = self.Save["Efield"]
            SaveB = self.Save["Bfield"]
            SaveA = self.Save["Angles"]
            SaveP = self.Save["Path"]
            SaveC = self.Save["Clock"]
            SaveT = self.Save["Energy"]

            Q = self.Particles[self.index].Z * Constants.e
            M = self.Particles[self.index].M
            E = self.Particles[self.index].E
            T = self.Particles[self.index].T
            r = np.array(self.Particles[self.index].coordinates)

            V_normalized = np.array(self.Particles[self.index].velocities)
            V_norm = Constants.c * np.sqrt(E ** 2 - M ** 2) / E
            Vm = V_norm * V_normalized

            q = self.Step * Q / 2 / (M * Units.MeV2kg)

            Step = self.Step
            Num = self.Num
            Nsave = self.Nsave
            i_save = 0
            st = timer()
            for i in range(Num):
                PathLen = V_norm * Step

                Vp, Yp, Ya, B, E = self.AlgoStep(T, M, q, Vm, r)
                Vm, T = self.RadLossStep(Vp, Vm, Yp, Ya, M, Q)
                V_norm, r_new = self.Update(PathLen, Step, TotPathLen, TotTime, Vm, r)
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

            print(f"Event No {self.index+1} of {len(self.Particles)} in {timer() - st} seconds")
            Saves = Saves[:i_save]
            Saves[:, :3] /= self.Bfield.ToMeters

            ret = Saves[:, :6]

            if SaveE:
                ret = np.hstack((ret, Saves[:, 6:9]))
            if SaveB:
                ret = np.hstack((ret, Saves[:, 9:12]))
            if SaveA:
                ret = np.hstack((ret, Saves[:, 12]))
            if SaveP:
                ret = np.hstack((ret, Saves[:, 13]))
            if SaveC:
                ret = np.hstack((ret, Saves[:, 14]))
            if SaveE:
                ret = np.hstack((ret, Saves[:, 15]))
            RetArr.append(ret)
        RetArr = np.array(RetArr)
        return RetArr

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def Update(PathLen, Step, TotPathLen, TotTime, Vm, r):
        V_norm = np.linalg.norm(Vm)
        r_new = r + Vm * Step
        TotTime += Step
        TotPathLen += PathLen
        return V_norm, r_new

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
