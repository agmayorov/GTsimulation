import numpy as np
from numba import jit

from gtsimulation import GTSimulator
from gtsimulation.Global import Constants

class HigueraCarySimulator(GTSimulator):
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
        u = Yp * V
        epsilon = q * E
        u_minus = u + epsilon
        u_minus_sq = np.dot(u_minus, u_minus)
        gamma_minus = np.sqrt(1.0 + u_minus_sq / (c * c))

        if H_norm == 0:
            u_f = u + 2.0 * epsilon
            gamma_new = np.sqrt(1.0 + np.dot(u_f, u_f) / (c * c))
            v_new = u_f / gamma_new
            gamma_avg = 0.5 * (Yp + gamma_new)
            return v_new, gamma_new, gamma_avg

        beta = q * H
        beta_sq = np.dot(beta, beta)

        beta_dot_u_minus = np.dot(beta, u_minus)
        term = gamma_minus * gamma_minus - beta_sq
        sqrt_arg = term * term + 4.0 * (beta_sq + (beta_dot_u_minus * beta_dot_u_minus) / (c * c))
        sqrt_term = np.sqrt(sqrt_arg)
        gamma_new_sq = 0.5 * (term + sqrt_term)
        gamma_new = np.sqrt(gamma_new_sq)

        t_vec = beta / gamma_new
        t_sq = np.dot(t_vec, t_vec)
        sigma = 2.0 * t_vec / (1.0 + t_sq)
        u_minus_cross_t = np.cross(u_minus, t_vec)
        u_plus = u_minus + np.cross(u_minus + u_minus_cross_t, sigma)

        u_f = u_plus + epsilon
        gamma_final = np.sqrt(1 + np.dot(u_f, u_f) / (c * c))
        v_new = u_f / gamma_final
        gamma_avg = 0.5 * (Yp + gamma_final)

        return v_new, gamma_final, gamma_avg