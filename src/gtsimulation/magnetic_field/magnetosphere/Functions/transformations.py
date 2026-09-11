import numpy as np
from numba import njit


@njit(fastmath=True)
def DirectionEarthtoSun(Year, Day, Secs):
    RAD = 57.2958
    GST = 0
    SLONG = 0
    SDEC = 0
    SRASN = 0
    if Year > 1901 or Year < 2099:
        FDAY = Secs / 86400
        DJ = 365 * (Year - 1900) + (Year - 1901) / 4 + Day + FDAY - 0.5
        T = DJ / 36525
        VL = np.mod(279.696678 + 0.9856473354 * DJ, 360)
        GST = np.mod(279.690983 + 0.9856473354 * DJ + 360 * FDAY + 180, 360)
        G = np.mod(358.475845 + 0.985600267 * DJ, 360) / RAD
        SLONG = VL + (1.91946 - 0.004789 * T) * np.sin(G) + 0.020094 * np.sin(2 * G)
        OBLIQ = (23.45229 - 0.0130125 * T) / RAD
        SLP = (SLONG - 0.005686) / RAD
        sinDD = np.sin(OBLIQ) * np.sin(SLP)
        cosDD = np.sqrt(1 - sinDD ** 2)
        SDEC = RAD * np.arctan(sinDD / cosDD)
        SRASN = 180 - RAD * np.arctan2(np.tan(OBLIQ) ** (-1) * sinDD / cosDD, -np.cos(SLP) / cosDD)
    Ex = np.cos(SRASN * np.pi / 180.) * np.cos(SDEC * np.pi / 180.)
    Ey = np.sin(SRASN * np.pi / 180.) * np.cos(SDEC * np.pi / 180.)
    Ez = np.sin(SDEC * np.pi / 180.)
    Vector = np.array([Ex, Ey, Ez])

    return Vector, GST, SLONG


@njit(fastmath=True)
def DisplacementForEccentricDipole(g, h):
    v = np.array([g[0, 0], g[0, 1], h[0, 1]])
    B0 = np.linalg.norm(v)
    L0 = 2 * g[0, 0] * g[1, 0] + np.sqrt(3) * (g[0, 1] * g[1, 1] + h[0, 1] * h[1, 1])
    L1 = -g[0, 1] * g[1, 0] + np.sqrt(3) * (g[0, 0] * g[1, 1] + g[0, 1] * g[1, 2] + h[0, 1] * h[1, 2])
    L2 = -h[0, 1] * g[1, 0] + np.sqrt(3) * (g[0, 0] * h[1, 1] - h[0, 1] * g[1, 2] + g[0, 1] * h[1, 2])
    E = (L0 * g[0, 0] + L1 * g[0, 1] + L2 * h[0, 1]) / (4 * B0 ** 2)
    DX = (L1 - g[0, 1] * E) / (3 * B0 ** 2)
    DY = (L2 - h[0, 1] * E) / (3 * B0 ** 2)
    DZ = (L0 - g[0, 0] * E) / (3 * B0 ** 2)

    return np.array([[DX], [DY], [DZ]])


@njit(fastmath=True)
def geo2mag_eccentric(x, y, z, g, h, inverse=False):
    RE = 6378137.1
    A = DisplacementForEccentricDipole(g, h) * RE

    # if np.ndim(x) > 1 and x.shape[1] == 1:
    #     x = np.transpose(x)
    #     y = np.transpose(y)
    #     z = np.transpose(z)

    mat = np.array([[0.339067758413505, -0.919633920274268, -0.198258689306225],
                      [0.938257039240758, 0.345938908356903, 0],
                      [0.068589929661063, -0.186019809236783, 0.980148994857721]])

    if not inverse:
        vec = mat @ (np.array([[x], [y], [z]], dtype=mat.dtype) - A)
    else:
        vec = np.transpose(mat) @ np.array([[x], [y], [z]], dtype=mat.dtype) + A

    vec = np.transpose(vec)

    X = vec[0, 0]
    Y = vec[0, 1]
    Z = vec[0, 2]

    return X, Y, Z

@njit(fastmath=True)
def gei2geo(x, y, z, year, doy, ut_sec, inverse=False):
    _, GST, _ = DirectionEarthtoSun(year, doy, ut_sec)
    theta = np.radians(GST)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if not inverse:
        # GEI → GEO
        X = cos_theta * x + sin_theta * y
        Y = -sin_theta * x + cos_theta * y
    else:
        # GEO → GEI
        X = cos_theta * x - sin_theta * y
        Y = sin_theta * x + cos_theta * y

    return X, Y, z


@njit(fastmath=True)
def gei2gsm(x, y, z, year, doy, ut_sec, g, h, inverse=False):
    if not inverse:
        # GEI -> GEO
        x, y, z = gei2geo(x, y, z, year, doy, ut_sec)

        # GEO -> GSM
        X, Y, Z = geo2gsm(x, y, z, year, doy, ut_sec, g, h)

    else:
        # GSM -> GEO
        x, y, z = geo2gsm(x, y, z, year, doy, ut_sec, g, h, inverse=True)

        # GEO -> GEI
        X, Y, Z = gei2geo(x, y, z, year, doy, ut_sec, inverse=True)

    return X, Y, Z

@njit(fastmath=True)
def geo2dipmag(x, y, z, psi, inverse=False):
    mat = np.array([[np.cos(psi), 0., np.sin(psi)],
                    [0., 1., 0.],
                    [-np.sin(psi), 0., np.cos(psi)]])
    if not inverse:
        vec = mat @ np.array([[x], [y], [z]], dtype=mat.dtype)
    else:
        vec = np.transpose(mat) @ np.array([[x], [y], [z]], dtype=mat.dtype)

    X = vec[0, 0]
    Y = vec[1, 0]
    Z = vec[2, 0]

    return X, Y, Z

@njit(fastmath=True)
def geo2mag(x, y, z, inverse=False):
    R = np.array([[0.339067758413505, -0.919633920274268, -0.198258689306225],
                    [0.938257039240758, 0.345938908356903, 0.],
                    [0.068589929661063, -0.186019809236783, 0.980148994857721]])

    if not inverse > 0:
        vec = R @ np.array([[x], [y], [z]], dtype=R.dtype)
    else:
        vec = R.T @ np.array([[x], [y], [z]], dtype=R.dtype)

    X = vec[0, 0]
    Y = vec[1, 0]
    Z = vec[2, 0]

    return X, Y, Z


@njit(fastmath=True)
def get_dipole_direction(g, h):
    g10 = g[0, 0]
    g11 = g[0, 1]
    h11 = h[0, 1]

    b0 = np.sqrt(g10**2 + g11**2 + h11**2)

    alpha = np.arccos(-g10 / b0)
    beta = np.arctan(h11 / g11)

    return np.array([
        np.sin(alpha) * np.cos(beta),
        np.sin(alpha) * np.sin(beta),
        np.cos(alpha)
    ])


@njit(fastmath=True)
def geo2gsm(x, y, z, year, doy, ut_sec, g, h, inverse=False):
    # Sun direction and Greenwich sidereal time
    sun, GST, _ = DirectionEarthtoSun(year, doy, ut_sec)

    xS = sun[0]
    yS = sun[1]
    zS = sun[2]

    theta = np.radians(GST)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Dipole direction in GEO
    dipole = get_dipole_direction(g, h)

    xDip = dipole[0]
    yDip = dipole[1]
    zDip = dipole[2]

    # Dipole direction in GEI
    xD = cos_theta * xDip - sin_theta * yDip
    yD = sin_theta * xDip + cos_theta * yDip
    zD = zDip

    # GSM Y axis
    xSD = yD * zS - yS * zD
    ySD = zD * xS - zS * xD
    zSD = xD * yS - xS * yD

    norm = np.sqrt(xSD * xSD + ySD * ySD + zSD * zSD)

    xSD /= norm
    ySD /= norm
    zSD /= norm

    # GSM Z axis
    xSSD = yS * zSD - ySD * zS
    ySSD = zS * xSD - zSD * xS
    zSSD = xS * ySD - xSD * yS

    norm = np.sqrt(
        xSSD * xSSD +
        ySSD * ySSD +
        zSSD * zSSD
    )

    xSSD /= norm
    ySSD /= norm
    zSSD /= norm

    if not inverse:
        # GEO -> GEI
        xGEI = cos_theta * x - sin_theta * y
        yGEI = sin_theta * x + cos_theta * y
        zGEI = z

        # GEI -> GSM
        X = xS * xGEI + yS * yGEI + zS * zGEI
        Y = xSD * xGEI + ySD * yGEI + zSD * zGEI
        Z = xSSD * xGEI + ySSD * yGEI + zSSD * zGEI

    else:
        # GSM -> GEI
        xGEI = xS * x + xSD * y + xSSD * z
        yGEI = yS * x + ySD * y + ySSD * z
        zGEI = zS * x + zSD * y + zSSD * z

        # GEI -> GEO
        X = cos_theta * xGEI + sin_theta * yGEI
        Y = -sin_theta * xGEI + cos_theta * yGEI
        Z = zGEI

    return X, Y, Z
