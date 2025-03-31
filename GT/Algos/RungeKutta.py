import numpy as np
from numba import jit

from GT import GTSimulator
from Global import Constants


class RungeKutta4Simulator(GTSimulator):
    def AlgoStep(self, T, M, q, V, X, H1, E):
        x, y, z = X
        vx, vy, vz = V
        dt = self.Step
        if self.Bfield is not None:
            # H1 = np.array(self.Bfield.GetBfield(x, y, z))
            H2 = np.array(self.Bfield.GetBfield(x + vx * dt / 2, y + vy * dt / 2, z + vz * dt / 2))
            H3 = np.array(self.Bfield.GetBfield(x + vx * dt, y + vy * dt, z + vz * dt))
            if len(H1.shape) == 2:
                # H1 = H1[:, 0]
                H2 = H2[:, 0]
                H3 = H3[:, 0]
        else:
            # H1 = np.zeros(3)
            H2 = np.zeros(3)
            H3 = np.zeros(3)

        # if self.Efield is not None:
        #     E = np.array(self.Efield.GetEfield(x, y, z))
        # else:
        #     E = np.zeros(3)

        if M != 0:
            return self.__algo(H1, H2, H3, M, T, V, q, dt)#, H1, E
        else:
            return V, 0, 0#, H1, E

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __algo(H1, H2, H3, M, T, V, q, dt):
        Yp = T / M + 1
        p = 2 * q / (Yp * dt)

        k1 = p * np.cross(V, H1)
        k2 = p * np.cross(V + dt / 2 * k1, H2)
        k3 = p * np.cross(V + dt / 2 * k2, H2)
        k4 = p * np.cross(V + dt * k3, H3)

        Vp = dt / 6 * (k1 + 2 * (k2 + k3) + k4) + V

        return Vp, Yp, Yp


class RungeKutta6Simulator(GTSimulator):
    def AlgoStep(self, T, M, q, V, X, H1, E):
        x, y, z = X
        vx, vy, vz = V
        dt = self.Step
        c = np.array([0, 1 / 3, 2 / 3, 1 / 3, 1 / 2, 1 / 2, 1])
        b = np.array([11 / 120, 0, 27 / 40, 27 / 40, -4 / 15, -4 / 15, 11 / 120])
        a = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [1 / 3, 0, 0, 0, 0, 0, 0],
                      [0, 2 / 3, 0, 0, 0, 0, 0],
                      [1 / 12, 1 / 3, -1 / 12, 0, 0, 0, 0],
                      [-1 / 16, 9 / 8, -3 / 16, -3 / 8, 0, 0, 0],
                      [0, 9 / 8, -3 / 8, -3 / 4, 1 / 2, 0, 0],
                      [9 / 44, -9 / 11, 63 / 44, 18 / 11, 0, -16 / 11, 0]])
        if self.Bfield is not None:
            # H1 = np.array(self.Bfield.GetBfield(x, y, z))
            H2 = np.array(self.Bfield.GetBfield(x + vx * dt * c[1], y + vy * dt * c[1], z + vz * dt * c[1]))
            H3 = np.array(self.Bfield.GetBfield(x + vx * dt * c[2], y + vy * dt * c[2], z + vz * dt * c[2]))
            H4 = H2
            H5 = np.array(self.Bfield.GetBfield(x + vx * dt * c[4], y + vy * dt * c[4], z + vz * dt * c[4]))
            H6 = H5
            H7 = np.array(self.Bfield.GetBfield(x + vx * dt * c[6], y + vy * dt * c[6], z + vz * dt * c[6]))
            if len(H1.shape) == 2:
                H1 = H1[:, 0]
                H2 = H2[:, 0]
                H3 = H3[:, 0]
                H4 = H4[:, 0]
                H5 = H5[:, 0]
                H6 = H6[:, 0]
                H7 = H7[:, 0]
        else:
            # H1 = np.zeros(3)
            H2 = np.zeros(3)
            H3 = np.zeros(3)
            H4 = np.zeros(3)
            H5 = np.zeros(3)
            H6 = np.zeros(3)
            H7 = np.zeros(3)

        # if self.Efield is not None:
        #     E = np.array(self.Efield.GetEfield(x, y, z))
        # else:
        #     E = np.zeros(3)
        if M != 0:
            return self.__algo(H1, H2, H3, H4, H5, H6, H7, a, b, M, T, V, q, dt)#, H1, E
        else:
            return V, 0, 0#, H1, E

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __algo(H1, H2, H3, H4, H5, H6, H7, a, b, M, T, V, q, dt):
        Yp = T / M + 1
        p = 2 * q / (Yp * dt)

        k = np.zeros((7, 3))

        k[0] = p * np.cross(V, H1)
        k[1] = p * np.cross(V + dt * k[0] * a[1, 0], H2)
        k[2] = p * np.cross(V + dt * k[1] * a[2, 1], H3)
        k[3] = p * np.cross(V + dt * (a[3, 0] * k[0] + a[3, 1] * k[1] + a[3, 2] * k[2]), H4)
        k[4] = p * np.cross(V + dt * (a[4, 0] * k[0] + a[4, 1] * k[1] + a[4, 2] * k[2] + a[4, 3] * k[3]), H5)
        k[5] = p * np.cross(V + dt * (a[5, 0] * k[0] + a[5, 1] * k[1] + a[5, 2] * k[2] + a[5, 3] * k[3] + a[5, 4]*k[4]),
                            H6)
        k[6] = p * np.cross(V + dt * (a[6, 0] * k[0] + a[6, 1] * k[1] + a[6, 2] * k[2] + a[6, 3] * k[3] + a[6, 4]*k[4] +
                                      a[6, 5]*k[5]), H7)

        Vp = dt * np.sum(b*k.T, axis=1) + V

        return Vp, Yp, Yp
