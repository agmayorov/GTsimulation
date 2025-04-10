import numpy as np
from numba import jit


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
