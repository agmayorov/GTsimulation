import numpy as np
from numba import jit

from GT import GTSimulator
from Global import Constants


class BunemanBorisSimulator(GTSimulator):
    def AlgoStep(self, T, M, q, V, X, H, E):
        x, y, z = X
        # if self.Bfield is not None:
        #     H = np.array(self.Bfield.GetBfield(x, y, z))
        #     if len(H.shape) == 2:
        #         H = H[:, 0]
        # else:
        #     H = np.zeros(3)
        #
        # if self.Efield is not None:
        #     E = np.array(self.Efield.GetEfield(x, y, z))
        # else:
        #     E = np.zeros(3)
        c = Constants.c
        if M != 0:
            return self.__algo(E, H, M, T, V, q, c) #, H, E
        else:
            return V, 0, 0#, H, E

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def __algo(E, H, M, T_particle, V, q, c):
        H_norm = np.linalg.norm(H)
        Yp = T_particle / M + 1
        if H_norm == 0:
            return V, Yp, Yp
        Ui = Yp * V

        TT = Yp * np.tan(q * H_norm / Yp)

        T = TT * H / H_norm

        U = np.cross(V, T) + 2 * q * E + Ui

        UU = (np.dot(U, T)) ** 2 / c ** 2
        YY = np.sqrt(1 + np.linalg.norm(U) ** 2 / c ** 2)

        S = YY ** 2 - TT ** 2

        Ym = Yp
        Yp = np.sqrt(0.5 * (S + np.sqrt(S ** 2 + 4 * (TT ** 2 + UU))))
        Ya = 0.5 * (Ym + Yp)

        tt = np.tan(q * H_norm / Yp)

        t = tt * H / H_norm

        s = 1 / (1 + tt ** 2)

        Vp = s / Yp * (U + t * np.dot(U, t) + np.cross(U, t))

        return Vp, Yp, Ya
        # return Vp, Yp, Ya
