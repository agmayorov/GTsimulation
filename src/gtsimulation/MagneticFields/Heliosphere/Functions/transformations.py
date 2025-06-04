import numpy as np
from numba import jit


@jit(fastmath=True, nopython=True)
def Cart2Sphere(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-12)
    R = np.sqrt(x ** 2 + y ** 2 + 1e-12)
    theta = np.arccos(z / r)
    phi = np.arccos(x / R) * (y >= 0) + (2 * np.pi - np.arccos(x / R)) * (y < 0)
    return r, R, theta, phi


@jit(fastmath=True, nopython=True)
def Sphere2Cart(Br, Btheta, Bphi, theta, phi):
    Bx = Br * np.sin(theta) * np.cos(phi) + Btheta * np.cos(theta) * np.cos(phi) - Bphi * np.sin(phi)
    By = Br * np.sin(theta) * np.sin(phi) + Btheta * np.cos(theta) * np.sin(phi) + Bphi * np.cos(phi)
    Bz = Br * np.cos(theta) - Btheta * np.sin(theta)
    return Bx, By, Bz
