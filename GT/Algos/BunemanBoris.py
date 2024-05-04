import numpy as np

from GT import GTSimulator, Constants


class BunemanBorisSimulator(GTSimulator):
    def AlgoStep(self, T, M, q, V, X):
        x, y, z = X
        Vx, Vy, Vz = V
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

        Hx, Hy, Hz = H
        Ex, Ey, Ez = E

        # st = timer()

        H_norm = np.linalg.norm(H)

        Yp = T / M + 1
        Uix = Yp * Vx
        Uiy = Yp * Vy
        Uiz = Yp * Vz

        TT = Yp * np.tan(q * H_norm / Yp)

        Tx = TT * Hx / H_norm
        Ty = TT * Hy / H_norm
        Tz = TT * Hz / H_norm

        Ux = Vy * Tz - Vz * Ty + 2 * q * Ex + Uix
        Uy = Vz * Tx - Vx * Tz + 2 * q * Ey + Uiy
        Uz = Vx * Ty - Vy * Tx + 2 * q * Ez + Uiz

        UU = (Tx * Ux + Ty * Uy + Tz * Uz) ** 2 / Constants.c ** 2
        YY = np.sqrt(1 + (Ux ** 2 + Uy ** 2 + Uz ** 2) / Constants.c ** 2)

        S = YY ** 2 - TT ** 2

        Ym = Yp
        Yp = np.sqrt(0.5 * (S + np.sqrt(S ** 2 + 4 * (TT ** 2 + UU))))
        Ya = 0.5 * (Ym + Yp)

        tt = np.tan(q * H_norm / Yp)

        tx = tt * Hx / H_norm
        ty = tt * Hy / H_norm
        tz = tt * Hz / H_norm

        s = 1 / (1 + tt ** 2)

        Vpx = s / Yp * (Ux+(Ux*tx+Uy*ty+Uz*tz)*tx+(Uy*tz-Uz*ty))
        Vpy = s / Yp * (Uy+(Ux*tx+Uy*ty+Uz*tz)*ty+(Uz*tx-Ux*tz))
        Vpz = s / Yp * (Uz+(Ux*tx+Uy*ty+Uz*tz)*tz+(Ux*ty-Uy*tx))

        Vp = np.array([Vpx, Vpy, Vpz])

        return Vp, Yp, Ya
