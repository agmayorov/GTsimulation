import numpy as np
from abc import ABC, abstractmethod

import tqdm

from GT import Constants, Units


class GTSimulator(ABC):
    def __init__(self, *args, **kwargs):
        self.Bfield = None
        self.Efield = None
        self.Medium = None
        self.Save = None
        self.Step = None
        self.Num = None
        self.Particles = None
        self.Date = None
        self.UseRadLosses = False
        self.BackTracing = False
        self.index = 0

    def __call__(self):
        for self.index in range(len(self.Particles)):
            TotTime, TotPathLen = 0, 0
            for _ in tqdm.tqdm(range(self.Num)):
                E = self.Particles[self.index].E
                M = self.Particles[self.index].M
                T = self.Particles[self.index].T

                r = np.array(self.Particles[self.index].coordinates)

                V_normalized = np.array(self.Particles[self.index].velocities)
                V_norm = Constants.c * np.sqrt(E ** 2 - M ** 2) / (T + M)
                Vm = V_norm * V_normalized
                PathLen = V_norm * self.Step

                Q = self.Particles[self.index].Q

                self.SimulationStep(M, T, Vm, Q, r)

                TotTime += self.Step
                TotPathLen += PathLen

    def SimulationStep(self, M, T, Vm, Q, r):
        q = self.Step * Q / 2 / (M * Units.MeV2kg)
        Vp, Yp, Ya = self.AlgoStep(T, M, q, Vm, r)
        Vm, T = self.RadLossStep(Vp, Vm, Yp, Ya, M, Q)
        self.Particles[self.index].UpdateState(Vm, T, self.Step)

    def RadLossStep(self, Vp, Vm, Yp, Ya, M, Q):
        if not self.UseRadLosses:
            T = M * (Yp - 1)
            return Vm, T

        acc = (Vp - Vm) / self.Step
        Vn = np.linalg.norm(Vp + Vm)
        Vinter = (Vp + Vm) / Vn

        acc_par = np.dot(acc, Vinter)
        acc_per = np.sqrt(np.linalg.norm(acc) ** 2 - acc_par ** 2)

        dE = self.Step * ((2 / (3 * 4 * np.pi * 8.854187e-12) * Q ** 2 * Ya ** 4 / Constants.c ** 3) *
                          (acc_per ** 2 + acc_par ** 2 * Ya ** 2) / Constants.e / 1e6)

        RadLossBT = 1 if not self.BackTracing else -1

        T = M * (Yp - 1) - RadLossBT * np.abs(dE)

        V = Constants.c * np.sqrt((T + M) ** 2 - M ** 2) / (T + M)
        Vn = np.linalg.norm(Vp)

        Vm = V * Vp / Vn

        return Vm, T

    @abstractmethod
    def AlgoStep(self, T, M, q, Vm, r):
        pass
