import datetime

import numpy as np
from numba import jit
from scipy.io import loadmat
import os


# @jit(fastmath=True, nopython=True)
def vecRotMat(f, t):
    # assert np.linalg.norm(f) == 1 and np.linalg.norm(t) == 1
    #
    # f = np.array(f)
    # t = np.array(t)

    if np.all(f == t):
        return np.diag([1, 1, 1])
    elif np.all(f == -t):
        return np.diag([-1, -1, -1])

    v = np.cross(f, t)
    s = np.linalg.norm(v)
    c = np.dot(f, t)
    v_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.diag([1, 1, 1]) + v_mat + v_mat @ v_mat * (1 - c) / s ** 2


def load_pamela(path, d, ev):
    exposure = loadmat(path + os.sep + "Exposure_" + str(d) + ".mat", variable_names=['Date', 'lat', 'lon', 'alt'])
    event = loadmat(path + os.sep + 'NTrack_' + str(d) + '.mat')["Event"] - 1
    Sij = loadmat(path + os.sep + "PitchAngles_" + str(d) + ".mat")["Sij"][ev, :]
    Rig = loadmat(path + os.sep + 'Tracker_' + str(d) + '.mat')["Rig"][ev, :]

    Date = exposure["Date"][event[ev], :].astype(int)

    Date = datetime.datetime(year=Date[0, 0], month=Date[0, 1], day=Date[0, 2], hour=Date[0, 3], minute=Date[0, 4],
                             second=Date[0, 5])


    lat = exposure["lat"][event[ev], :]
    lon = exposure["lon"][event[ev], :]
    alt = exposure["alt"][event[ev], :]

    return Date, lat, lon, alt, Sij, Rig




if __name__ == "__main__":
    load_pamela(r"../Data/PAML3", 212, 5)