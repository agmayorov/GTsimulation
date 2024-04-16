import datetime
import numpy as np

from MagneticFields.Magnetosphere.Functions import gauss


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
    Vector = [Ex, Ey, Ez]

    return Vector, GST, SLONG


def DisplacementForEccentricDipole(date: datetime.datetime):
    g, h, _ = gauss.LoadGaussCoeffs("MagneticFields/Magnetosphere/IGRF13/igrf13coeffs.npy", date)
    B0 = np.linalg.norm([g[0, 0], g[0, 1], h[0, 1]])
    L0 = 2 * g[0, 0] * g[1, 0] + np.sqrt(3) * (g[0, 1] * g[1, 1] + h[0, 1] * h[1, 1])
    L1 = -g[0, 1] * g[1, 0] + np.sqrt(3) * (g[0, 0] * g[1, 1] + g[0, 1] * g[1, 2] + h[0, 1] * h[1, 2])
    L2 = -h[0, 1] * g[1, 0] + np.sqrt(3) * (g[0, 0] * h[1, 1] - h[0, 1] * g[1, 2] + g[0, 1] * h[1, 2])
    E = (L0 * g[0, 0] + L1 * g[0, 1] + L2 * h[0, 1]) / (4 * B0 ** 2)
    DX = (L1 - g[0, 1] * E) / (3 * B0 ** 2)
    DY = (L2 - h[0, 1] * E) / (3 * B0 ** 2)
    DZ = (L0 - g[0, 0] * E) / (3 * B0 ** 2)

    return np.array([[DX], [DY], [DZ]])


def geo2mag_eccentric(x, y, z, j, date: datetime.datetime):
    RE = 6378137.1
    A = DisplacementForEccentricDipole(date) * RE

    if np.ndim(x) > 1 and x.shape[1] == 1:
        x = np.transpose(x)
        y = np.transpose(y)
        z = np.transpose(z)

    if j > 0:
        vec = np.dot([[0.339067758413505, -0.919633920274268, -0.198258689306225],
                      [0.938257039240758, 0.345938908356903, 0],
                      [0.068589929661063, -0.186019809236783, 0.980148994857721]],
                     (np.vstack((x, y, z)) - A))
    else:
        vec = np.dot(np.transpose([[0.339067758413505, -0.919633920274268, -0.198258689306225],
                                   [0.938257039240758, 0.345938908356903, 0],
                                   [0.068589929661063, -0.186019809236783, 0.980148994857721]]),
                     np.vstack((x, y, z))) + A

    vec = np.transpose(vec)

    X = vec[:, 0][:, np.newaxis]
    Y = vec[:, 1][:, np.newaxis]
    Z = vec[:, 2][:, np.newaxis]

    return X, Y, Z


def gei2geo(x, y, z, Year, Day, Secs, j):
    _, GST, _ = DirectionEarthtoSun(Year, Day, Secs)
    if j > 0:
        vec = np.transpose([[np.cos(np.radians(GST)), -np.sin(np.radians(GST)), 0],
                            [np.sin(np.radians(GST)), np.cos(np.radians(GST)), 0],
                            [0, 0, 1]]) @ np.vstack((x, y, z))
    else:
        vec = [[np.cos(np.radians(GST)), -np.sin(np.radians(GST)), 0],
               [np.sin(np.radians(GST)), np.cos(np.radians(GST)), 0],
               [0, 0, 1]] @ np.vstack((x, y, z))

    X = vec[0, :]
    Y = vec[1, :]
    Z = vec[2, :]

    return X, Y, Z


def gei2gsm(x, y, z, Year, Day, Secs, j):
    S, GST, _ = DirectionEarthtoSun(Year, Day, Secs)

    D = np.array([[0.068589929661063], [-0.186019809236783], [0.980148994857721]])

    D = np.array([[np.cos(np.radians(GST)), -np.sin(np.radians(GST)), 0],
                  [np.sin(np.radians(GST)), np.cos(np.radians(GST)), 0],
                  [0, 0, 1]]) @ D

    a = np.cross(D[:, 0], S)
    Y = a / np.sqrt(np.sum(a ** 2))

    Z = np.cross(S, Y)

    if j > 0:
        vec = np.vstack((S, Y, Z)) @ np.vstack((x, y, z))
    else:
        vec = np.vstack((S, Y, Z)).T @ np.vstack((x, y, z))

    X = vec[0, :]
    Y = vec[1, :]
    Z = vec[2, :]

    return X, Y, Z


def geo2dipmag(x, y, z, psi, j):
    if j > 0:
        vec = np.array([[np.cos(np.radians(psi)), 0, np.sin(np.radians(psi))],
                        [0, 1, 0],
                        [-np.sin(np.radians(psi)), 0, np.cos(np.radians(psi))]]) @ np.vstack((x, y, z))
    else:
        vec = np.array([[np.cos(np.radians(psi)), 0, np.sin(np.radians(psi))],
                        [0, 1, 0],
                        [-np.sin(np.radians(psi)), 0, np.cos(np.radians(psi))]]).T @ np.vstack((x, y, z))

    X = vec[0, :]
    Y = vec[1, :]
    Z = vec[2, :]

    return X, Y, Z


def geo2mag(x, y, z, j):
    if j > 0:
        vec = np.array([[0.339067758413505, -0.919633920274268, -0.198258689306225],
                        [0.938257039240758, 0.345938908356903, 0],
                        [0.068589929661063, -0.186019809236783, 0.980148994857721]]) @ np.vstack((x, y, z))
    else:
        vec = np.array([[0.339067758413505, -0.919633920274268, -0.198258689306225],
                        [0.938257039240758, 0.345938908356903, 0],
                        [0.068589929661063, -0.186019809236783, 0.980148994857721]]).T @ np.vstack((x, y, z))

    X = vec[0, :]
    Y = vec[1, :]
    Z = vec[2, :]

    return X, Y, Z


def geo2gsm(x, y, z, Year, DoY, Secs, d):
    S, GST, _ = DirectionEarthtoSun(Year, DoY, Secs)
    D = np.dot([[np.cos(np.radians(GST)), -np.sin(np.radians(GST)), 0],
                [np.sin(np.radians(GST)), np.cos(np.radians(GST)), 0],
                [0, 0, 1]], np.array([0.068589929661063, -0.186019809236783, 0.980148994857721]))
    a = np.cross(D, S)
    Y = a / np.sqrt(np.sum(a ** 2))
    Z = np.cross(S, Y)

    if np.ndim(x) > 1 and x.shape[1] == 1:
        x = np.transpose(x)
        y = np.transpose(y)
        z = np.transpose(z)

    if d == 1:
        vec = np.vstack((S, Y, Z)) @ (np.array([[np.cos(np.radians(GST)), -np.sin(np.radians(GST)), 0],
                                               [np.sin(np.radians(GST)), np.cos(np.radians(GST)), 0],
                                               [0, 0, 1]]) @ np.vstack((x, y, z)))
    else:
        vec = np.array([[np.cos(np.radians(GST)), -np.sin(np.radians(GST)), 0],
                        [np.sin(np.radians(GST)), np.cos(np.radians(GST)), 0],
                        [0, 0, 1]]).T @ (np.vstack((S, Y, Z)).T @ np.vstack((x, y, z)))

    vec = vec.T

    X = vec[:, 0][:, np.newaxis]
    Y = vec[:, 1][:, np.newaxis]
    Z = vec[:, 2][:, np.newaxis]

    return X, Y, Z
