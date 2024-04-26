import numpy as np

from consts import Constants, Units


class GTSimulator:
    def __init__(self, *args, **kwargs):
        self.Bfield = None
        self.Efield = None
        self.Medium = None
        self.Save = None
        self.Step = None
        self.Particles = None
        self.Date = None
        self.UseRadLosses = False
        self.index = 0

    def __call__(self):
        for self.index in range(len(self.Particles)):
            E = self.Particles[self.index].E
            M = self.Particles[self.index].M
            T = self.Particles[self.index].T

            r = np.array(self.Particles[self.index].coordinates)

            V_normalized = np.array(self.Particles[self.index].velocities)
            V_norm = Constants.c * np.sqrt(E ** 2 - M ** 2) / (T + M)
            Vm = V_norm * V_normalized

            Q = self.Particles[self.index].Q
            q = self.Step * Q / 2 / (M * Units.MeV2kg)

            Vp, Yp, Ya = self.Buneman_Boris_step(T, M, q, Vm, r)

            Vm, T = self.CalcRadLosses(Vp, Vm, Yp, Ya, M)
            self.Particles[self.index].UpdateState(Vm, T, self.Step)

    def CalcRadLosses(self, Vp, Vm, Yp, Ya, M):
        if not self.UseRadLosses:
            T = M * (Yp - 1)
            return Vm, T

        # TODO add radiation losses

    def Buneman_Boris_step(self, T, M, q, V, X):
        x, y, z = X
        if self.Bfield is not None:
            H = np.array(self.Bfield.GetBfield(x, y, z))
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

        S = YY**2 - TT**2

        Ym = Yp
        Yp = np.sqrt(0.5*(S + np.sqrt(S**2 + 4 * (TT**2 + UU))))
        Ya = 0.5*(Ym + Yp)

        tt = np.tan(q*H_norm/Yp)

        t = tt * H/H_norm

        s = 1/(1 + tt*2)

        Vp = s/Yp * (U + t*np.dot(U, t) + np.cross(U, t))

        return Vp, Yp, Ya



