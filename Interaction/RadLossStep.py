# @jit(nopython=True, fastmath=True)

import numpy as np
from .SynchrotronEmission import *
from .GenSynchCounter import SynchCounter


def MakeRadLossStep(Vp, Vm, Yp, Ya, M, Q, rm, self, particle, Constants, synch_record: SynchCounter):
    new_photons = []

    if not self.UseRadLosses[0]:
        T = M * (Yp - 1)
        return Vp, T, new_photons, synch_record

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

    if self.UseRadLosses[1]:
        T, B_perp = synch_record.get_averages()
        N_avg = get_N_avg(B_perp, synch_record.delta_t, M, particle.Z)
        if N_avg > 1000:
            E_keV_photons = MakeSynchrotronEmission(synch_record.delta_t, T, B_perp, M, particle.Z)
            E_MeV_photons = E_keV_photons * 1e-3
            E_MeV_photons = E_MeV_photons[(E_MeV_photons >= self.UseRadLosses[2][0]) & (E_MeV_photons <= self.UseRadLosses[2][1])]
            for Energy in E_MeV_photons:
                new_photon = {"Track": {"Coordinates": rm, "Velocities": Vm},
                              "Particle": {"PDG": 22, "M": 0, "T": Energy, "Gen": self.__gen + 1}}
                new_photons.append(new_photon)

    return Vm, T, new_photons, synch_record