import numpy as np
from numba import jit

from gtsimulation import GTSimulator
from gtsimulation.common import Constants, Units


class VaySimulator(GTSimulator):
    def AlgoStep(self, T, M, Q, V, X, H, E):
        dt = self.Step

        # Half position step:
        # X^(n+1/2) = X^n + v^n * dt / 2
        X_half = X + V * dt / 2
        E_half, H_half = self.getFields(X_half)

        if M != 0:
            q = dt * Q / (2 * M * Units.MeV2kg)

            # Velocity update
            Vp, Yp, Ya = self.__algo(
                E_half, H_half, M, T, V, q, Constants.c
            )
        else:
            Vp, Yp, Ya = V, 0, 0

        # Second half position step:
        # X^(n+1) = X^(n+1/2) + v^(n+1) * dt / 2
        X_new = X_half + Vp * dt / 2

        return X_new, Vp, Yp, Ya


    def getFields(self, X):
        x, y, z = X
        if self.Bfield is not None:
            H = np.array(self.Bfield.GetBfield(x, y, z))
        else:
            H = np.zeros(3)

        if self.Efield is not None:
            E = np.array(self.Efield.GetEfield(x, y, z))
        else:
            E = np.zeros(3)

        return E, H

    @staticmethod
    @jit(fastmath=True, nopython=True)
    def __algo(E, H, M, T_particle, V, q, c):
        H_norm = np.linalg.norm(H)
        Y0 = T_particle / M + 1
        if H_norm == 0 and np.linalg.norm(E) == 0:
            return V, Y0, Y0

        Ui = Y0 * V

        Ui_hs = Ui + q * (E + np.cross(V, H))

        U_ = Ui_hs + q * E

        T = q * H_norm

        T_v = T * H / H_norm if H_norm > 0 else np.zeros(3)

        U = (np.dot(U_, T_v)) / c

        Y = np.sqrt(1 + np.linalg.norm(U_) ** 2 / c ** 2)

        S = Y ** 2 - T ** 2

        Y_s = np.sqrt(0.5 * (S + np.sqrt(S ** 2 + 4 * (T ** 2 + U ** 2))))
        Y_hs = 0.5 * (Y0 + Y_s)

        t = T_v / Y_s

        s = 1 / (1 + np.linalg.norm(t) ** 2)

        V_s = (s * (U_ + t * np.dot(U_, t) + np.cross(U_, t))) / Y_s

        return V_s, Y_s, Y_hs