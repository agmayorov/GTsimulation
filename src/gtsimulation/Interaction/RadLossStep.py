# @jit(nopython=True, fastmath=True)

import numpy as np
from .SynchrotronEmission import *
from .GenSynchCounter import SynchCounter


def MakeRadLossStep(Vp, Vm, Yp, Ya, M, Q, rm, dt, frwdTracing, sync_params, particle, Gen, Constants,
                    synch_record: SynchCounter):
    new_photons = []

    acc = (Vp - Vm) / dt
    Vn = np.linalg.norm(Vp + Vm)
    Vinter = (Vp + Vm) / Vn

    acc_par = np.dot(acc, Vinter)
    acc_per = np.sqrt(np.linalg.norm(acc) ** 2 - acc_par ** 2)

    dE = dt * ((2 / (3 * 4 * np.pi * 8.854187e-12) * Q ** 2 * Ya ** 4 / Constants.c ** 3) *
                      (acc_per ** 2 + acc_par ** 2 * Ya ** 2) / Constants.e / 1e6)

    T = M * (Yp - 1) - frwdTracing * np.abs(dE)

    V = Constants.c * np.sqrt((T + M) ** 2 - M ** 2) / (T + M)
    Vn = np.linalg.norm(Vp)

    Vm = V * Vp / Vn

    if sync_params[0]:
        T, B_perp = synch_record.get_averages()
        N_avg = get_N_avg(B_perp, synch_record.delta_t, M, particle.Z)
        if N_avg > 1000:
            E_keV_photons = MakeSynchrotronEmission(synch_record.delta_t, T, B_perp, M, particle.Z)
            E_MeV_photons = E_keV_photons * 1e-3
            E_MeV_photons = E_MeV_photons[(E_MeV_photons >= sync_params[1][0]) & (E_MeV_photons <= sync_params[1][1])]
            for Energy in E_MeV_photons:
                new_photon = {"Track": {"Coordinates": rm, "Velocities": Vm/np.linalg.norm(Vm)},
                              "Particle": {"PDG": 22, "M": 0, "T": Energy, "Gen": Gen + 1}}
                new_photons.append(new_photon)
            synch_record.reset()

    return Vm, T, new_photons, synch_record
