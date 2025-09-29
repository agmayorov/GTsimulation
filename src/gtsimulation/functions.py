import numpy as np
from numba import jit

def GetLastPoints(RetArr_i, s):
    R = RetArr_i["Track"]['Coordinates'][-1]
    V = RetArr_i["Track"]["Velocities"][-1]
    if s == -1:
        V *= -1
    return R, V

@jit(fastmath=True, nopython=True)
def CalcPitchAngles(H: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    H: ndarray of float [nT], shape(3,) or (N, 3) - magnetic field vectors
    V: ndarray of float [m/s], shape(3,) or (N, 3) - velocity vectors

    Returns
    -------
    PitchAngles: ndarray [degrees], shape() or (N,) - pitch angles in degrees
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

@jit(fastmath=True, nopython=True)
def CalcLarmorRadii(Hm: np.ndarray, T: float, pitchd: float, M: float, Z: int) -> np.ndarray:
    """
    Parameters
    ----------
    Hm: ndarray of float [T], shape(N,) - module of magnetic induction Bm
    T: float [MeV] - kinetic energy
    pitchd: float [degree] - pitch angle
    M: float [MeV] - mass
    Z: int [p+] - charge

    Returns
    -------
    larmor: ndarray of float [m], shape(N,) - larmor radius

    Used formulas
    -------------
    p = np.sqrt((T+M)**2 - M**2)
    r = p * sin(pitch) / (q * B)
    """
    Z = abs(Z)
    cc = 2.99792458e8

    larmor = (np.sqrt((T + M)**2 - M**2) * 1e6 / cc * np.sin(pitchd/180 * np.pi)) / (Z * Hm)

    return larmor

@jit(fastmath=True, nopython=True)
def CalcGuidingCenter(coo: np.ndarray,
                      V: np.ndarray,
                      H: np.ndarray,
                      T: float,
                      pitch_deg: float,
                      M: float,
                      Z: int) -> np.ndarray:
    """
    Parameters
    ----------
    coo: ndarray [m], shape(m,3)
    V: ndarray, shape(m,3)
    H: ndarray [nT], shape(m,3)
    T: float
    pitch_deg: float
    M [MeV]
    Z [p+]

    Returns
    -------
    center [m]
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