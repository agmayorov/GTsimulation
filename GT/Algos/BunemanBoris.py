import numpy as np

from GT import GTSimulator, Constants


class BunemanBorisSimulator(GTSimulator):
    def AlgoStep(self, T, M, q, V, X):
        x, y, z = X
        if self.Bfield is not None:
            H = np.array(self.Bfield.GetBfield(x, y, z))
            if len(H.shape) == 2:
                H = H[:, 0]
        else:
            H = np.zeros(3)

        if self.Efield is not None:
            E = np.array(self.Efield.GetEfield(x, y, z))
        else:
            E = np.zeros(3)

        H_norm = np.linalg.norm(H)

        Yp = T / M + 1
        Ui = Yp * V

        TT = Yp * np.tan(q * H_norm / Yp)

        T = TT * H / H_norm

        U = np.cross(V, T) + 2 * q * E + Ui

        UU = (np.dot(U, T)) ** 2 / Constants.c ** 2
        YY = np.sqrt(1 + np.linalg.norm(U) ** 2 / Constants.c ** 2)

        S = YY ** 2 - TT ** 2

        Ym = Yp
        Yp = np.sqrt(0.5 * (S + np.sqrt(S ** 2 + 4 * (TT ** 2 + UU))))
        Ya = 0.5 * (Ym + Yp)

        tt = np.tan(q * H_norm / Yp)

        t = tt * H / H_norm

        s = 1 / (1 + tt ** 2)

        Vp = s / Yp * (U + t * np.dot(U, t) + np.cross(U, t))

        return Vp, Yp, Ya
