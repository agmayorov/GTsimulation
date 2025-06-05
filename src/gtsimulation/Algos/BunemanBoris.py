import numpy as np
from numba import jit

from gtsimulation.GT import GTSimulator
from gtsimulation.Global import Constants


class BunemanBorisSimulator(GTSimulator):
    def AlgoStep(self, T, M, q, V, X, H, E):
        c = Constants.c
        if M != 0:
            return self.__algo(E, H, M, T, V, q, c)
        else:
            return V, 0, 0

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def __algo(E, H, M, T_particle, V, q, c):
        H_norm = np.linalg.norm(H)
        Yp = T_particle / M + 1
        if H_norm == 0 and np.linalg.norm(E) == 0:
            return V, Yp, Yp
        Ui = Yp * V

        TT = Yp * np.tan(q * H_norm / Yp)

        T = TT * H / H_norm if H_norm > 0 else np.zeros(3)

        U = np.cross(V, T) + 2 * q * E + Ui

        UU = (np.dot(U, T)) ** 2 / c ** 2
        YY = np.sqrt(1 + np.linalg.norm(U) ** 2 / c ** 2)

        S = YY ** 2 - TT ** 2

        Ym = Yp
        Yp = np.sqrt(0.5 * (S + np.sqrt(S ** 2 + 4 * (TT ** 2 + UU))))
        Ya = 0.5 * (Ym + Yp)

        tt = np.tan(q * H_norm / Yp)

        t = tt * H / H_norm if H_norm > 0 else np.zeros(3)

        s = 1 / (1 + tt ** 2)

        Vp = s / Yp * (U + t * np.dot(U, t) + np.cross(U, t))

        return Vp, Yp, Ya
