import numpy as np


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
        cosDD = np.sqrt(1 - sinDD ^ 2)
        SDEC = RAD * np.atan(sinDD / cosDD)
        SRASN = 180 - RAD * np.arctan2(np.cot(OBLIQ) * sinDD / cosDD, -np.cos(SLP) / cosDD)
    Ex = np.cos(SRASN*np.pi / 180.) * np.cos(SDEC*np.pi / 180.)
    Ey = np.sin(SRASN*np.pi / 180.) * np.cos(SDEC*np.pi / 180.)
    Ez = np.sin(SDEC*np.pi / 180.)
    Vector = [Ex, Ey, Ez]

    return Vector, GST, SLONG
