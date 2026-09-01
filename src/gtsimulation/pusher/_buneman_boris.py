import numpy as np
from numba import njit

from gtsimulation import GTSimulator
from gtsimulation.common import Constants, Units


class BunemanBorisSimulator(GTSimulator):
    """
    Relativistic Buneman-Boris particle pusher.

    Advances the relativistic motion of charged particles in electric and
    magnetic fields using the Buneman-Boris scheme. The particle proper
    velocity is advanced through a symmetric splitting of the electric-field
    acceleration and magnetic-field rotation, followed by conversion to the
    ordinary velocity.

    Notes
    -----
    The Buneman-Boris pusher is based on the Boris particle-pushing scheme,
    which employs a time-reversible, leap-frog formulation to advance the
    particle motion. The time-reversible difference procedures used in the
    scheme are closely related to those introduced by Buneman [1]_, who
    considered, among other examples, the Lorentz equation and relativistic
    particle orbits. The relativistic formulation of the particle pusher was
    subsequently presented by Boris [2]_.

    In a purely magnetic field, the magnetic part of the scheme corresponds
    to a rotation of the particle proper velocity and therefore preserves its
    magnitude exactly, so that the particle kinetic energy is conserved
    exactly apart from floating-point round-off.

    The implementation used here follows the formulation given by Ripperda
    et al. [3]_, where the relativistic Boris pusher is described in a form
    suitable for numerical implementation.

    References
    ----------
    .. [1] Buneman, O. "Time-reversible difference procedures."
       Journal of Computational Physics 1.4 (1967): 517-535.
    .. [2] Boris, J. P. "Relativistic plasma simulation — optimization of a
       hybrid code." Proceedings of the 4th Conference on Numerical
       Simulation of Plasmas (1970): 3-67.
    .. [3] Ripperda, B., et al. "A Comprehensive Comparison of
       Relativistic Particle Integrators." The Astrophysical Journal
       Supplement Series 235.1 (2018): 21.
    """
    def AlgoStep(self, T, M, Q, V, X, H, E):
        if M != 0:
            q = self.Step * Q / (2 * M * Units.MeV2kg)
            Vp, Yp, Ya = self.__algo(E, H, M, T, V, q, Constants.c)
        else:
            Vp, Yp, Ya = V, 0, 0
        X_new = X + Vp * self.Step
        return X_new, Vp, Yp, Ya

    @staticmethod
    @njit(fastmath=True)
    def __algo(E, H, M, T_particle, V, q, c):
        # Initial gamma and proper velocity U = gamma * V
        gamma0 = T_particle / M + 1.0
        U = gamma0 * V
        # Boris: first half-push by electric field
        U += q * E
        # Gamma after first half E-push
        gamma_minus = np.sqrt(1.0 + np.dot(U, U) / c**2)
        # Magnetic rotation
        ## t = (q dt / 2m) B / gamma_minus
        t = q * H / gamma_minus
        ## s = 2t / (1 + t^2)
        s = 2.0 * t / (1.0 + np.dot(t, t))
        ## u+ = u- + [(u- + [u- x t]) x s]
        U = U + np.cross(U + np.cross(U, t), s)
        # Second half-push by electric field
        U += q * E
        # Final gamma
        gamma_plus = np.sqrt(1.0 + np.dot(U, U) / c**2)
        # Return ordinary velocity
        V_new = U / gamma_plus
        return V_new, gamma_plus, gamma_minus
