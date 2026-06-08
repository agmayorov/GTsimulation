import numpy as np
from numba import jit

from gtsimulation import GTSimulator
from gtsimulation.Global import Constants, Units

class RungeKutta4Simulator(GTSimulator):
    # def getF(self, X, V, Q):
    #     x, y, z = X
    #     if self.Bfield is not None:
    #         H = np.array(self.Bfield.GetBfield(x, y, z))
    #     else:
    #         H = np.zeros(3)
    #
    #     if self.Efield is not None:
    #         E = np.array(self.Efield.GetEfield(x, y, z))
    #     else:
    #         E = np.zeros(3)
    #
    #     return Q * (E + np.cross(V, H))
    #
    # def getf(self, XP_vec, Q, m):
    #     X = XP_vec[:3]
    #     P = XP_vec[-3:]
    #     Y = np.sqrt(1 + (np.linalg.norm(P) / (m * Constants.c)) ** 2)
    #     V = P / Y / m
    #     F = self.getF(X, V, Q)
    #     return np.concatenate(V, F)
    #
    # def RK4Pusher(self, X, P, Q, m, dt):
    #     if m == 0:
    #         raise ValueError("RK4Pusher does not support the case M = 0.")
    #
    #     XP_vec = np.concatenate(X, P)
    #
    #     k1 = self.getf(XP_vec, Q, m)
    #     k2 = self.getf(XP_vec + 0.5 * dt * k1, Q, m)
    #     k3 = self.getf(XP_vec + 0.5 * dt * k2, Q, m)
    #     k4 = self.getf(XP_vec + dt * k3, Q, m)
    #
    #     XP_vec_new = XP_vec + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    #     X_new = XP_vec_new[:3]
    #     P_new = XP_vec_new[-3:]
    #
    #     return X_new, P_new

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

    def AlgoStep(self, T, M, Q, V, X, H, E):
        if M != 0:
            m = M * Units.MeV2kg
            Ym = T / M + 1
            P = Ym * m * V
            dt = self.Step
            c = Constants.c

            # k1
            V1, F1 = self.__algo(P, Q, m, c, E, H)

            # k2
            X2 = X + 0.5 * dt * V1
            P2 = P + 0.5 * dt * F1
            E2, H2 = self.getFields(X2)
            V2, F2 = self.__algo(P2, Q, m, c, E2, H2)

            # k3
            X3 = X + 0.5 * dt * V2
            P3 = P + 0.5 * dt * F2
            E3, H3 = self.getFields(X3)
            V3, F3 = self.__algo(P3, Q, m, c, E3, H3)

            # k4
            X4 = X + dt * V3
            P4 = P + dt * F3
            E4, H4 = self.getFields(X4)
            V4, F4 = self.__algo(P4, Q, m, c, E4, H4)

            X_new = X + dt * (V1 + 2 * V2 + 2 * V3 + V4) / 6
            P_new = P + dt * (F1 + 2 * F2 + 2 * F3 + F4) / 6

            Y_new = np.sqrt(1 + (np.linalg.norm(P_new) / (m * c)) ** 2)
            V_new = P_new / Y_new / m
            Ya = 0.5 * (Ym + Y_new)

        else:
            V_new, Y_new, Ya = V, 0, 0

        return X_new, V_new, Y_new, Ya

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __algo(P, Q, m, c, E, H):
        Y = np.sqrt(1 + (np.linalg.norm(P) / (m * c)) ** 2)
        V = P / Y / m
        F = Q * (E + np.cross(V, H))
        return V, F

class RungeKutta4SimulatorFast(GTSimulator):
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

    def AlgoStep(self, T, M, Q, V, X, H, E):
        if M != 0:
            m = M * Units.MeV2kg
            Ym = T / M + 1
            P = Ym * m * V
            dt = self.Step
            c = Constants.c

            # k1
            P1 = P
            V1, F1 = self.__algo(P1, Q, m, E, H, c)

            # k2
            P2 = P + 0.5 * dt * F1
            V2, F2 = self.__algo(P2, Q, m, E, H, c)

            # k3
            P3 = P + 0.5 * dt * F2
            V3, F3 = self.__algo(P3, Q, m, E, H, c)

            # k4
            P4 = P + dt * F3
            V4, F4 = self.__algo(P4, Q, m, E, H, c)

            X_new, V_new, Y_new, Ya = self.__algo2(X, P, dt, m, c, Ym, V1, V2, V3, V4, F1, F2, F3, F4)


        else:
            V_new, Y_new, Ya = V, 0, 0

        return X_new, V_new, Y_new, Ya

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __algo(P, Q, m, E, H, c):
        Y = np.sqrt(1 + (np.linalg.norm(P) / (m * c)) ** 2)
        V = P / Y / m
        F = Q * (E + np.cross(V, H))
        return V, F

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def __algo2(X, P, dt, m, c, Ym, V1, V2, V3, V4, F1, F2, F3, F4):
        X_new = X + dt * (V1 + 2 * V2 + 2 * V3 + V4) / 6
        P_new = P + dt * (F1 + 2 * F2 + 2 * F3 + F4) / 6

        Y_new = np.sqrt(1 + (np.linalg.norm(P_new) / (m * c)) ** 2)
        V_new = P_new / Y_new / m
        Ya = 0.5 * (Ym + Y_new)
        return X_new, V_new, Y_new, Ya


# class RungeKutta4Simulator(GTSimulator):
#     def AlgoStep(self, T, M, q, V, X, H1, E):
#         x, y, z = X
#         vx, vy, vz = V
#         dt = self.Step
#         if self.Bfield is not None:
#             # H1 = np.array(self.Bfield.GetBfield(x, y, z))
#             H2 = np.array(self.Bfield.GetBfield(x + vx * dt / 2, y + vy * dt / 2, z + vz * dt / 2))
#             H3 = np.array(self.Bfield.GetBfield(x + vx * dt, y + vy * dt, z + vz * dt))
#             if len(H1.shape) == 2:
#                 # H1 = H1[:, 0]
#                 H2 = H2[:, 0]
#                 H3 = H3[:, 0]
#         else:
#             # H1 = np.zeros(3)
#             H2 = np.zeros(3)
#             H3 = np.zeros(3)
#
#         # if self.Efield is not None:
#         #     E = np.array(self.Efield.GetEfield(x, y, z))
#         # else:
#         #     E = np.zeros(3)
#
#         if M != 0:
#             return self.__algo(H1, H2, H3, M, T, V, q, dt)#, H1, E
#         else:
#             return V, 0, 0#, H1, E
#
#     @staticmethod
#     @jit(nopython=True, fastmath=True)
#     def __algo(H1, H2, H3, M, T, V, q, dt):
#         Yp = T / M + 1
#         p = 2 * q / (Yp * dt)
#
#         k1 = p * np.cross(V, H1)
#         k2 = p * np.cross(V + dt / 2 * k1, H2)
#         k3 = p * np.cross(V + dt / 2 * k2, H2)
#         k4 = p * np.cross(V + dt * k3, H3)
#
#         Vp = dt / 6 * (k1 + 2 * (k2 + k3) + k4) + V
#
#         return Vp, Yp, Yp


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
