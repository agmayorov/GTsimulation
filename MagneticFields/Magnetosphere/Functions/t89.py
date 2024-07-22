"""
Copied from https://github.com/tsssss/geopack/blob/master/geopack/t89.py
"""

import numpy as np
from numba import jit


@jit(fastmath=True, nopython=True)
def t89(iopt, ps, x, y, z):
    """
    Computes GSM components of the magnetic field produced by extra-
    terrestrial current systems in the geomagnetosphere. The model is
    valid up to geocentric distances of 70 Re and is based on the merged
    IMP-A,C,D,E,F,G,H,I,J (1966-1974), HEOS-1 and -2 (1969-1974),
    and ISEE-1 and -2  spacecraft data set.

    This is a modified version (t89c), which replaced the original one
    in 1992 and differs from it in the following:
    (1) ISEE-1,2 data were added to the original IMP-HEOS dataset
    (2) Two terms were added to the original tail field modes, allowing
        a modulation of the current by the geodipole tilt angle

    :param iopt: specifies the ground disturbance level:
        iopt= 1       2        3        4        5        6      7
                   correspond to:
        kp=  0,0+  1-,1,1+  2-,2,2+  3-,3,3+  4-,4,4+  5-,5,5+  &gt =6-
    :param ps: geo-dipole tilt angle in radius.
    :param x,y,z: GSM coordinates in Re (1 Re = 6371.2 km)
    :return: bx,by,bz. Field components in GSM system, in nT.

    Reference for the original model: N.A. Tsyganenko, A magnetospheric magnetic
        field model with a warped tail current sheet: planet.space sci., v.37, pp.5-20, 1989.

    This release of t89c is dated  Feb 12, 1996;
    Last update: May 9, 2006; a save statement was added in the subroutine t89c, to avoid runtime problems on some fortran compilers
    Author: Nikolai A. Tsyganenko, HSTX Corp./NASA GSFC
    """
    param = np.array([[-1.1653e+02, -5.5553e+01, -1.0134e+02, -1.8169e+02, -4.3654e+02,
                       -7.0777e+02, -1.1904e+03],
                      [-1.0719e+04, -1.3198e+04, -1.3480e+04, -1.2320e+04, -9.0010e+03,
                       -4.4719e+03, 2.7499e+03],
                      [4.2375e+01, 6.0647e+01, 1.1135e+02, 1.7379e+02, 3.2366e+02,
                       4.3281e+02, 7.4256e+02],
                      [5.9753e+01, 6.1072e+01, 1.2386e+01, -9.6664e+01, -4.1008e+02,
                       -4.3551e+02, -1.1103e+03],
                      [-1.1363e+04, -1.6064e+04, -2.4699e+04, -3.9051e+04, -5.0340e+04,
                       -6.0400e+04, -7.7193e+04],
                      [1.7844e+00, 2.2534e+00, 2.6459e+00, 3.2633e+00, 3.9932e+00,
                       4.6229e+00, 7.6727e+00],
                      [3.0268e+01, 3.4407e+01, 3.8948e+01, 4.4968e+01, 5.8524e+01,
                       6.8178e+01, 1.0205e+02],
                      [-3.5372e-02, -3.8887e-02, -3.4080e-02, -4.6377e-02, -3.8519e-02,
                       -8.8245e-02, -9.6015e-02],
                      [-6.6832e-02, -9.4571e-02, -1.2404e-01, -1.6686e-01, -2.6822e-01,
                       -2.1002e-01, -7.4507e-01],
                      [1.6456e-02, 2.7154e-02, 2.9702e-02, 4.8298e-02, 7.4528e-02,
                       1.1846e-01, 1.1214e-01],
                      [-1.3024e+00, -1.3901e+00, -1.4052e+00, -1.5473e+00, -1.4268e+00,
                       -2.6711e+00, -1.3614e+00],
                      [1.6529e-03, 1.3460e-03, 1.2103e-03, 1.0277e-03, -1.0985e-03,
                       2.2305e-03, 1.5157e-03],
                      [2.0293e-03, 1.3238e-03, 1.6381e-03, 3.1632e-03, 9.6613e-03,
                       1.0910e-02, 2.2283e-02],
                      [2.0289e+01, 2.3005e+01, 2.4490e+01, 2.7341e+01, 2.7557e+01,
                       2.7547e+01, 2.3164e+01],
                      [-2.5203e-02, -3.0565e-02, -3.7705e-02, -5.0655e-02, -5.6522e-02,
                       -5.4080e-02, -7.4146e-02],
                      [2.2491e+02, 5.5047e+01, -2.9832e+02, -5.1410e+02, -8.6703e+02,
                       -4.2423e+02, -2.2191e+03],
                      [-9.2348e+03, -3.8757e+03, 4.4009e+03, 1.2482e+04, 2.0652e+04,
                       1.1002e+03, 4.8253e+04],
                      [2.2788e+01, 2.0178e+01, 1.8692e+01, 1.6257e+01, 1.4101e+01,
                       1.3954e+01, 1.2714e+01],
                      [7.8813e+00, 7.9693e+00, 7.9064e+00, 8.5834e+00, 8.3501e+00,
                       7.5337e+00, 7.6777e+00],
                      [1.8362e+00, 1.4575e+00, 1.3047e+00, 1.0194e+00, 7.2996e-01,
                       8.9714e-01, 5.7138e-01],
                      [-2.7228e-01, 8.9471e-01, 2.4541e+00, 3.6148e+00, 3.8149e+00,
                       3.7813e+00, 2.9633e+00],
                      [8.8184e+00, 9.4039e+00, 9.7012e+00, 8.6042e+00, 9.2908e+00,
                       8.2945e+00, 9.3909e+00],
                      [2.8714e+00, 3.5215e+00, 7.1624e+00, 5.5057e+00, 6.4674e+00,
                       5.1740e+00, 9.7263e+00],
                      [1.4468e+01, 1.4474e+01, 1.4288e+01, 1.3778e+01, 1.3729e+01,
                       1.4213e+01, 1.1123e+01],
                      [3.2177e+01, 3.6555e+01, 3.3822e+01, 3.2373e+01, 2.8353e+01,
                       2.5237e+01, 2.1558e+01],
                      [1.0000e-02, 1.0000e-02, 1.0000e-02, 1.0000e-02, 1.0000e-02,
                       1.0000e-02, 1.0000e-02],
                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                       0.0000e+00, 0.0000e+00],
                      [7.0459e+00, 7.0787e+00, 6.7442e+00, 7.3195e+00, 7.4237e+00,
                       7.0037e+00, 4.4518e+00],
                      [4.0000e+00, 4.0000e+00, 4.0000e+00, 4.0000e+00, 4.0000e+00,
                       4.0000e+00, 4.0000e+00],
                      [2.0000e+01, 2.0000e+01, 2.0000e+01, 2.0000e+01, 2.0000e+01,
                       2.0000e+01, 2.0000e+01]])

    id = 1
    a = param[:, int(iopt) - 1]
    xi = np.array([x, y, z, ps])
    bx, by, bz, der = extern(id, a, xi)
    return bx, by, bz


@jit(fastmath=True, nopython=True)
def extern(id, a, xi):
    """
    Calculates dependent model variables and their derivatives
    for given independent variables and model parameters.
    Specifies model functions with free parameters which
    must be determined by means of least squares fits (RMS
    minimization procedure).

    :param id: number of the data point in a set (initial assignments are performed
        only for ID=1, saving thus CPU time)
    :param a: input vector containing model parameters
    :param xi: input vector containing independent variables
    :return: fx,fy,fz. Vector containing calculated values of dependent variables;
        der. Vector containing calculated values for derivatives
            of dependent variables with respect to model parameters;

    T89 represents external magnetospheric magnetic field in Cartesian SOLAR MAGNETOSPHERIC coordinates
    (Tsyganenko N.A., Planet. Space Sci., 1989, v.37, p.5-20; the "T89 model" with the warped
    tail current sheet) + a modification added in April 1992 (see below)

    Model formulas for the magnetic field components contain in total
    30 free parameters (17 linear and 13 nonlinear parameters).

    Linear parameters:
        a[0]-a[1]: correspond to contribution from the tail current system
        a[2]-a[3]: the amplitudes of symmetric and antisymmetric terms in the contribution from the closure currents
        a[4]: the ring current amplitude
        a[5]-a[14]: define Chapman-Ferraro+Birkeland current field.
            The coefficients c16-c19  (see Formula 20 in the original paper), due to DivB=0
            condition, are expressed through a[5]-a[14] and hence are not independent ones.
        a[15]-a[16]: the terms which yield the tilt angle dependence of the tail current intensity (added on April 9, 1992)

    Nonlinear parameters:
        a[17]: dx - characteristic scale of the Chapman-Ferraro field along the x-axis
        a[18]: ADR (aRC) - Characteristic radius of the ring current
        a[19] : D0 - Basic half-thickness of the tail current sheet
        a[20] : DD (GamRC)- defines rate of thickening of the ring current, as we go from night- to dayside
        a[21] : Rc - an analog of "hinging distance" entering formula (11)
        a[22] : G - amplitude of tail current warping in the Y-direction
        a[23] : aT - Characteristic radius of the tail current
        a[24] : Dy - characteristic scale distance in the Y direction entering in W(x,y) in (13)
        a[25] : Delta - defines the rate of thickening of the tail current sheet in the Y-direction (in T89 it was fixed at 0.01)
        a[26] : Q - this parameter was fixed at 0 in the final version of T89; initially it was introduced for making Dy to depend on X
        a[27] : Sx (Xo) - enters in W(x,y) ; see (13)
        a[28] : Gam (GamT) - enters in DT in (13) and defines rate of tail sheet thickening on going from night to dayside; in T89 fixed at 4.0
        a[29] : Dyc - the Dy parameter for closure current system; in T89 fixed at 20.0

    Author: N.A. Tsyganenko, Dec 8-10, 1991
    """

    # The last four quantities define variation of tail sheet thickness along X
    a02, xlw2, yn, rpi, rt = [25., 170., 30.0, 0.318309890, 30.]
    xd, xld2 = [0., 40.]

    # The two quantities belong to the function WC which confines tail closure current in X- and Y- direction
    sxc, xlwc2 = [4., 50.]

    dxl = 20.

    der = np.zeros((3, 30))
    dyc = a[29]  # Dyc - the Dy parameter for closure current system; in T89 fixed at 20.0
    dyc2 = dyc ** 2
    dx = a[17]  # characteristic scale of the Chapman-Ferraro field along the x-axis
    ha02 = 0.5 * a02
    rdx2m = -1. / dx ** 2
    rdx2 = -rdx2m
    rdyc2 = 1 / dyc2
    hlwc2m = -0.5 * xlwc2
    drdyc2 = -2 * rdyc2
    drdyc3 = 2 * rdyc2 * np.sqrt(rdyc2)
    hxlw2m = -0.5 * xlw2

    adr = a[18]  # ADR (aRC) - Characteristic radius of the ring current
    d0 = a[19]  # D0 - Basic half-thickness of the tail current sheet
    dd = a[20]  # DD (GamRC)- defines rate of thickening of the ring current, as we go from night- to dayside
    rc = a[21]  # Rc - an analog of "hinging distance" entering formula (11)
    g = a[22]  # G - amplitude of tail current warping in the Y-direction
    at = a[23]  # aT - Characteristic radius of the tail current
    dt = d0
    p = a[24]  # Dy - characteristic scale distance in the Y direction entering in W(x,y) in (13)
    delt = a[
        25]  # Delta - defines the rate of thickening of the tail current sheet in the Y-direction (in T89 it was fixed at 0.01)
    q = a[
        26]  # Q - this parameter was fixed at 0 in the final version of T89; initially it was introduced for making Dy to depend on X
    sx = a[27]  # Sx (Xo) - enters in W(x,y) ; see (13)
    gam = a[
        28]  # Gam (GamT) - enters in DT in (13) and defines rate of tail sheet thickening on going from night to dayside; in T89 fixed at 4.0
    hxld2m = -0.5 * xld2
    adsl, xghs, h, hs, gamh = [0.] * 5
    dbldel = 2. * delt
    w1 = -0.5 / dx
    w2 = w1 * 2.
    w4 = -1. / 3.
    w3 = w4 / dx
    w5 = -0.5
    w6 = -3.
    ak1, ak2, ak3, ak4, ak5, ak6, ak7, ak8, ak9, ak10, ak11, ak12, ak13, ak14, ak15, ak16, ak17 = a[0:17]
    sxa, sya, sza = [0.] * 3
    ak610 = ak6 * w1 + ak10 * w5
    ak711 = ak7 * w2 - ak11
    ak812 = ak8 * w2 + ak12 * w6
    ak913 = ak9 * w3 + ak13 * w4
    rdxl = 1. / dxl
    hrdxl = 0.5 * rdxl
    a6h = ak6 * 0.5
    a9t = ak9 / 3.
    ynp = rpi / yn * 0.5
    ynd = 2. * yn

    x, y, z, tilt = xi[0:4]
    tlt2 = tilt ** 2
    sps = np.sin(tilt)
    cps = np.cos(tilt)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    tps = sps / cps
    htp = tps * 0.5
    gsp = g * sps
    xsm = x * cps - z * sps
    zsm = x * sps + z * cps

    # calculate the function zs defining the shape of the tail current sheet and its spatial derivatives:
    xrc = xsm + rc
    xrc16 = xrc ** 2 + 16
    sxrc = np.sqrt(xrc16)
    y4 = y2 * y2
    y410 = y4 + 1e4
    sy4 = sps / y410
    gsy4 = g * sy4
    zs1 = htp * (xrc - sxrc)
    dzsx = -zs1 / sxrc
    zs = zs1 - gsy4 * y4
    d2zsgy = -sy4 / y410 * 4e4 * y2 * y
    dzsy = g * d2zsgy

    # calculate the components of the ring current contribution:
    xsm2 = xsm ** 2
    dsqt = np.sqrt(xsm2 + a02)
    fa0 = 0.5 * (1 + xsm / dsqt)
    ddr = d0 + dd * fa0
    dfa0 = ha02 / dsqt ** 3
    zr = zsm - zs
    tr = np.sqrt(zr ** 2 + ddr ** 2)
    rtr = 1 / tr
    ro2 = xsm2 + y2
    adrt = adr + tr
    adrt2 = adrt ** 2
    fk = 1 / (adrt2 + ro2)
    dsfc = np.sqrt(fk)
    fc = fk ** 2 * dsfc
    facxy = 3 * adrt * fc * rtr
    xzr = xsm * zr
    yzr = y * zr
    dbxdp = facxy * xzr
    der[1, 4] = facxy * yzr
    xzyz = xsm * dzsx + y * dzsy
    faq = zr * xzyz - ddr * dd * dfa0 * xsm
    dbzdp = fc * (2 * adrt2 - ro2) + facxy * faq
    der[0, 4] = dbxdp * cps + dbzdp * sps
    der[2, 4] = dbzdp * cps - dbxdp * sps

    # calculate the tail current sheet contribution:
    dely2 = delt * y2
    d = dt + dely2
    if np.abs(gam) >= 1e-6:
        xxd = xsm - xd
        rqd = 1 / (xxd ** 2 + xld2)
        rqds = np.sqrt(rqd)
        h = 0.5 * (1 + xxd * rqds)
        hs = -hxld2m * rqd * rqds
        gamh = gam * h
        d = d + gamh
        xghs = xsm * gam * hs
        adsl = -d * xghs
    d2 = d ** 2
    t = np.sqrt(zr ** 2 + d2)
    xsmx = xsm - sx
    rdsq2 = 1 / (xsmx ** 2 + xlw2)
    rdsq = np.sqrt(rdsq2)
    v = 0.5 * (1 - xsmx * rdsq)
    dvx = hxlw2m * rdsq * rdsq2
    om = np.sqrt(np.sqrt(xsm2 + 16) - xsm)
    oms = -om / (om * om + xsm) * 0.5
    rdy = 1 / (p + q * om)
    omsv = oms * v
    rdy2 = rdy ** 2
    fy = 1 / (1 + y2 * rdy2)
    w = v * fy
    yfy1 = 2 * fy * y2 * rdy2
    fypr = yfy1 * rdy
    fydy = fypr * fy
    dwx = dvx * fy + fydy * q * omsv
    ydwy = -v * yfy1 * fy
    ddy = dbldel * y
    att = at + t
    s1 = np.sqrt(att ** 2 + ro2)
    f5 = 1 / s1
    f7 = 1 / (s1 + att)
    f1 = f5 * f7
    f3 = f5 ** 3
    f9 = att * f3
    fs = zr * xzyz - d * y * ddy + adsl
    xdwx = xsm * dwx + ydwy
    rtt = 1 / t
    wt = w * rtt
    brrz1 = wt * f1
    brrz2 = wt * f3
    dbxc1 = brrz1 * xzr
    dbxc2 = brrz2 * xzr
    der[1, 0] = brrz1 * yzr
    der[1, 1] = brrz2 * yzr
    der[1, 15] = der[1, 0] * tlt2
    der[1, 16] = der[1, 1] * tlt2
    wtfs = wt * fs
    dbzc1 = w * f5 + xdwx * f7 + wtfs * f1
    dbzc2 = w * f9 + xdwx * f1 + wtfs * f3
    der[0, 0] = dbxc1 * cps + dbzc1 * sps
    der[0, 1] = dbxc2 * cps + dbzc2 * sps
    der[2, 0] = dbzc1 * cps - dbxc1 * sps
    der[2, 1] = dbzc2 * cps - dbxc2 * sps
    der[0, 15] = der[0, 0] * tlt2
    der[0, 16] = der[0, 1] * tlt2
    der[2, 15] = der[2, 0] * tlt2
    der[2, 16] = der[2, 1] * tlt2

    # calculate contribution from the closure currents
    zpl = z + rt
    zmn = z - rt
    rogsm2 = x2 + y2
    spl = np.sqrt(zpl ** 2 + rogsm2)
    smn = np.sqrt(zmn ** 2 + rogsm2)
    xsxc = x - sxc
    rqc2 = 1 / (xsxc ** 2 + xlwc2)
    rqc = np.sqrt(rqc2)
    fyc = 1 / (1 + y2 * rdyc2)
    wc = 0.5 * (1 - xsxc * rqc) * fyc
    dwcx = hlwc2m * rqc2 * rqc * fyc
    dwcy = drdyc2 * wc * fyc * y
    szrp = 1 / (spl + zpl)
    szrm = 1 / (smn - zmn)
    xywc = x * dwcx + y * dwcy
    wcsp = wc / spl
    wcsm = wc / smn
    fxyp = wcsp * szrp
    fxym = wcsm * szrm
    fxpl = x * fxyp
    fxmn = -x * fxym
    fypl = y * fxyp
    fymn = -y * fxym
    fzpl = wcsp + xywc * szrp
    fzmn = wcsm + xywc * szrm
    der[0, 2] = fxpl + fxmn
    der[0, 3] = (fxpl - fxmn) * sps
    der[1, 2] = fypl + fymn
    der[1, 3] = (fypl - fymn) * sps
    der[2, 2] = fzpl + fzmn
    der[2, 3] = (fzpl - fzmn) * sps

    # now calculate contribution from Chapman-Ferraro sources + all other
    ex = np.exp(x / dx)
    ec = ex * cps
    es = ex * sps
    ecz = ec * z
    esz = es * z
    eszy2 = esz * y2
    eszz2 = esz * z2
    ecz2 = ecz * z
    esy = es * y

    der[0, 5] = ecz
    der[0, 6] = es
    der[0, 7] = esy * y
    der[0, 8] = esz * z
    der[1, 9] = ecz * y
    der[1, 10] = esy
    der[1, 11] = esy * y2
    der[1, 12] = esy * z2
    der[2, 13] = ec
    der[2, 14] = ec * y2
    der[2, 5] = ecz2 * w1
    der[2, 9] = ecz2 * w5
    der[2, 6] = esz * w2
    der[2, 10] = -esz
    der[2, 7] = eszy2 * w2
    der[2, 11] = eszy2 * w6
    der[2, 8] = eszz2 * w3
    der[2, 12] = eszz2 * w4

    # finally, calculate net external magnetic field components, but first of all those for c.-f. field:
    sx1 = ak6 * der[0, 5] + ak7 * der[0, 6] + ak8 * der[0, 7] + ak9 * der[0, 8]
    sy1 = ak10 * der[1, 9] + ak11 * der[1, 10] + ak12 * der[1, 11] + ak13 * der[1, 12]
    sz1 = ak14 * der[2, 13] + ak15 * der[2, 14] + ak610 * ecz2 + ak711 * esz + ak812 * eszy2 + ak913 * eszz2
    bxcl = ak3 * der[0, 2] + ak4 * der[0, 3]
    bycl = ak3 * der[1, 2] + ak4 * der[1, 3]
    bzcl = ak3 * der[2, 2] + ak4 * der[2, 3]
    bxt = ak1 * der[0, 0] + ak2 * der[0, 1] + bxcl + ak16 * der[0, 15] + ak17 * der[0, 16]
    byt = ak1 * der[1, 0] + ak2 * der[1, 1] + bycl + ak16 * der[1, 15] + ak17 * der[1, 16]
    bzt = ak1 * der[2, 0] + ak2 * der[2, 1] + bzcl + ak16 * der[2, 15] + ak17 * der[2, 16]

    fx = bxt + ak5 * der[0, 4] + sx1 + sxa
    fy = byt + ak5 * der[1, 4] + sy1 + sya
    fz = bzt + ak5 * der[2, 4] + sz1 + sza

    return fx, fy, fz, der
