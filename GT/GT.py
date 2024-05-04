import matplotlib.pyplot as plt
import tqdm
import numpy as np
import importlib

from datetime import datetime
from abc import ABC, abstractmethod

from MagneticFields import Regions
from GT import Constants, Units


class GTSimulator(ABC):
    def __init__(self, Bfield=None, Efield=None, Region=Regions.Magnetosphere, Medium=None, Date=datetime(2008, 1, 1),
                 RadLosses=False, Particles="Monolines", ForwardTrck=None, Save=None, Num: int = 1e6, Step=1):
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

        self.Save = Save

        self.Step = Step
        self.Num = int(Num)

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

    def __call__(self):
        for self.index in range(len(self.Particles)):
            TotTime, TotPathLen = 0, 0
            Trajectory = []

            Q = self.Particles[self.index].Z * Constants.e
            M = self.Particles[self.index].M
            E = self.Particles[self.index].E
            T = self.Particles[self.index].T
            r = np.array(self.Particles[self.index].coordinates)

            V_normalized = np.array(self.Particles[self.index].velocities)
            V_norm = Constants.c * np.sqrt(E ** 2 - M ** 2) / E
            Vm = V_norm * V_normalized

            q = self.Step * Q / 2 / (M * Units.MeV2kg)

            for _ in tqdm.tqdm(range(self.Num)):
                Trajectory.append([r[0], r[1], r[2]])

                PathLen = V_norm * self.Step

                Vp, Yp, Ya = self.AlgoStep(T, M, q, Vm, r)
                Vm, T = self.RadLossStep(Vp, Vm, Yp, Ya, M, Q)
                # self.Particles[self.index].UpdateState(Vm, T, self.Step)
                r += Vm * self.Step

                TotTime += self.Step
                TotPathLen += PathLen

            Trajectory = np.array(Trajectory)
            Trajectory /= self.Bfield.ToMeters
            if self.Save is None:
                ax = plt.figure().add_subplot(projection='3d')
                ax.plot(0, 0, 0, '*')
                ax.plot(Trajectory[0, 0], Trajectory[0, 1], Trajectory[0, 2], 'o')
                ax.plot(Trajectory[-1, 0], Trajectory[-1, 1], Trajectory[-1, 2], 'o')
                ax.plot(Trajectory[:, 0], Trajectory[:, 1], Trajectory[:, 2])
                plt.show()

    # def SimulationStep(self, M, T, Vm, Q, r):
    #     q = self.Step * Q / 2 / (M * Units.MeV2kg)
    #     Vp, Yp, Ya = self.AlgoStep(T, M, q, Vm, r)
    #     Vm, T = self.RadLossStep(Vp, Vm, Yp, Ya, M, Q)
    #     self.Particles[self.index].UpdateState(Vm, T, self.Step)

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
