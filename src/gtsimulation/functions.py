import numpy as np
from numba import njit

def GetLastPoints(RetArr_i, s):
    R = RetArr_i["Track"]['Coordinates'][-1]
    V = RetArr_i["Track"]["Velocities"][-1]
    if s == -1:
        V *= -1
    return R, V


@njit(fastmath=True)
def CalcPitchAngles(H: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Calculate pitch angles of velocity vectors relative to the magnetic field.

    Parameters
    ----------
    H : numpy.ndarray
        Magnetic field vectors in nT. Shape ``(3,)`` or ``(N, 3)``.
    V : numpy.ndarray
        Velocity vectors in m/s. Shape ``(3,)`` or ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Pitch angles in degrees. Shape ``()`` or ``(N,)``.
    """

    H = np.asarray(H, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    H = np.atleast_2d(H)
    V = np.atleast_2d(V)

    # Compute norms manually
    Vn = np.sqrt(np.sum(V * V, axis=1))
    Hn = np.sqrt(np.sum(H * H, axis=1))

    VdotH = np.sum(V * H, axis=1)

    denominator = Vn * Hn
    cos_pitch = np.empty_like(VdotH)

    N = VdotH.shape[0]

    for i in range(N):
        if denominator[i] == 0.0:
            cos_pitch[i] = np.nan
        else:
            cos_pitch[i] = VdotH[i] / denominator[i]

    for i in range(N):
        if cos_pitch[i] > 1.0:
            cos_pitch[i] = 1.0
        elif cos_pitch[i] < -1.0:
            cos_pitch[i] = -1.0

    PitchAngles = np.arccos(cos_pitch) * (180.0 / np.pi)

    return PitchAngles


@njit(fastmath=True)
def CalcLarmorRadii(Hm: np.ndarray, T: float, pitchd: float, M: float, Z: int) -> np.ndarray:
    """
    Calculate Larmor radii of charged particles.

    Parameters
    ----------
    Hm : numpy.ndarray
        Magnetic induction magnitude in T. Shape ``(N,)``.
    T : float
        Kinetic energy in MeV.
    pitchd : float
        Pitch angle in degrees.
    M : float
        Particle mass in MeV/c².
    Z : int
        Particle charge number.

    Returns
    -------
    numpy.ndarray
        Larmor radius in m. Shape ``(N,)``.

    Notes
    -----
    The momentum and Larmor radius are calculated as

    ``p = np.sqrt((T + M)**2 - M**2)``

    ``r = p * sin(pitch) / (q * B)``
    """
    Z = abs(Z)
    cc = 2.99792458e8

    larmor = (np.sqrt((T + M)**2 - M**2) * 1e6 / cc * np.sin(pitchd/180 * np.pi)) / (Z * Hm)

    return larmor


@njit(fastmath=True)
def CalcGuidingCenter(coo: np.ndarray,
                      V: np.ndarray,
                      H: np.ndarray,
                      T: float,
                      pitch_deg: float,
                      M: float,
                      Z: int) -> np.ndarray:
    """
    Calculate the guiding center of a charged-particle trajectory.

    Parameters
    ----------
    coo : numpy.ndarray
        Particle coordinates in m. Shape ``(N, 3)``.
    V : numpy.ndarray
        Particle velocity vectors in m/s. Shape ``(N, 3)``.
    H : numpy.ndarray
        Magnetic field vectors in nT. Shape ``(N, 3)``.
    T : float
        Kinetic energy in MeV.
    pitch_deg : float
        Pitch angle in degrees.
    M : float
        Particle mass in MeV/c².
    Z : int
        Particle charge number.

    Returns
    -------
    numpy.ndarray
        Guiding-center coordinates in m.
    """
    H = np.asarray(H, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    H = np.atleast_2d(H)
    V = np.atleast_2d(V)

    Hm = np.sqrt(np.sum(H * H, axis=1))

    larm_radius = CalcLarmorRadii(Hm, T, pitch_deg, M, Z)
    VcrossH = np.cross(V, H)
    VcrossHn = np.sqrt(np.sum(VcrossH * VcrossH, axis=1))
    offset = np.sign(Z) * ((VcrossH.T / VcrossHn) * larm_radius).T
    center = coo + offset

    return center
