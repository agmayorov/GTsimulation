import numpy as np
from numba import jit, prange


@jit(fastmath=True, nopython=True)
def t15B(parmod, ps, x, y, z):
    pdpm = parmod[0]
    byimf = parmod[1]
    bzimf = parmod[2]
    xind = parmod[3]

    if xind > 2.0:
        print("WARNING: B-INDEX OUT OF ALLOWED RANGE")

    ps2 = ps ** 2

    a = np.array([
        1.,
        1.24658 + 0.336862 * xind - 0.151801 * xind ** 2,
        -0.715258E-01 - 0.164638 * xind - 0.367849E-01 * xind ** 2,
        0.927175 + 0.504440 * xind - 0.305463 * xind ** 2,
        -0.501315 - 2.41083 * xind + 1.57281 * xind ** 2,
        0.229589E-01 + 3.14061 * xind - 1.24458 * xind ** 2,
        -0.148657 - 0.595892 * xind + 0.330181 * xind ** 2,
        0.611957 + 2.22629 * xind - 0.693097 * xind ** 2,
        0.496858 - 1.63361 * xind + 0.405342 * xind ** 2,
        0.418382E-01 + 0.550920 * xind - 0.242575 * xind ** 2,
        0.251425 + 0.383031 * xind - 0.124622 * xind ** 2,
        0.351328 + 0.461355 * xind - 0.322270 * xind ** 2,
        0.551920 - 0.196532 * xind + 0.117334E-01 * xind ** 2,
        2.76976 - 2.25865 * xind + 1.23850 * xind ** 2,
        8.66404 + 1.22910 * xind - 1.38770 * xind ** 2,
        6.29907 + 1.20617 * xind - 0.535385 * xind ** 2,
        1.23683 + 0.921162 * xind - 0.307567 * xind ** 2,
        0.734181 + 0.149250 * xind - 0.139173 * xind ** 2,
        0.785365 + 0.944347 * xind - 0.377709 * xind ** 2,
        1.47682 - 3.23245 * xind + 1.35563 * xind ** 2,
        3.39086 - 0.918023 * xind + 0.361307 * xind ** 2,
        0.121136E-02 + 0.524103 * xind - 0.243464 * xind ** 2,
        0.367886 - 0.632629 * xind + 0.310636 * xind ** 2,
        3.50792 - 3.02391 * xind + 1.14657 * xind ** 2,
    ])

    parameters = np.array([
        a[1] + a[2]*ps2,
        a[3] + a[4]*ps2,
        a[5] + a[6]*ps2,
        a[7] + a[8]*ps2,
        a[9]*ps,
        a[10], a[11], a[12], a[13], a[14],
        a[15], a[16], a[17], a[18], a[19],
        a[20], a[21], a[22], a[23],
        xind, byimf, bzimf, pdpm, ps
    ])  # for any positive n index
        # fortran `PARAMETERS(n)` is our `parameters[n-2]`
        # fortran `PARAMETERS(-n)` is our `parameters[-n]`

    cfx, cfy, cfz = dipole_shield(x,y,z, ps, pdpm, bzimf)

    fpdpm = (pdpm/2.0)**(0.194)

    xx = x*fpdpm
    yy = y*fpdpm
    zz = z*fpdpm

    tbx, tby, tbz = deform_xz_yz(ps, parameters, xx, yy, zz, tail15_shielded)
    tx = parameters[0] * tbx
    ty = parameters[0] * tby
    tz = parameters[0] * tbz

    sbx, sby, sbz = deform_xz_yz(ps, parameters, xx, yy, zz, src_shielded)
    sx = parameters[1] * sbx
    sy = parameters[1] * sby
    sz = parameters[1] * sbz

    pbx, pby, pbz = deform_xz_yz(ps, parameters, xx, yy, zz, prc_shielded)
    px = parameters[2] * pbx
    py = parameters[2] * pby
    pz = parameters[2] * pbz

    theta00 = parameters[16]
    dtheta00 = parameters[17]

    bxr1r_unsh, byr1r_unsh, bzr1r_unsh, curdphi, dd, sp, cp, xxn, yyn, zzn, xxs, yys, zzs = r1_fac_r(theta00, dtheta00,
                                                                                                     ps, xx, yy, zz)

    bxr1r_shld, byr1r_shld, bzr1r_shld = r1_r_shld(xx, yy, zz, ps, bzimf, theta00, dtheta00)

    r1rx = parameters[3] * (bxr1r_unsh + bxr1r_shld) * fpdpm
    r1ry = parameters[3] * (byr1r_unsh + byr1r_shld) * fpdpm
    r1rz = parameters[3] * (bzr1r_unsh + bzr1r_shld) * fpdpm

    bxr1a_unsh, byr1a_unsh, bzr1a_unsh = r1_fac_a(ps, xx, yy, zz, curdphi, dd, sp, cp, xxn, yyn, zzn, xxs, yys, zzs)
    bxr1a_shld, byr1a_shld, bzr1a_shld = r1_fac_shld(xx, yy, zz, ps, bzimf, theta00, dtheta00)

    r1ax = parameters[4] * (bxr1a_unsh + bxr1a_shld) * fpdpm
    r1ay = parameters[4] * (byr1a_unsh + byr1a_shld) * fpdpm
    r1az = parameters[4] * (bzr1a_unsh + bzr1a_shld) * fpdpm

    bypen, bzpen = a[9] * byimf, a[9] * bzimf

    bx = cfx + tx + sx + px + r1rx + r1ax
    by = cfy + ty + sy + py + r1ry + r1ay + bypen
    bz = cfz + tz + sz + pz + r1rz + r1az + bzpen

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def tail15_shielded(x, y, z, parameters):
    pdpm = parameters[-2]
    bzimf = parameters[-3]

    pwr = parameters[6]
    pwp = parameters[7]
    xc = parameters[8]
    rn = parameters[9]
    rh = parameters[10]
    dmidn = parameters[18]

    t15fact = (pdpm/2.)**pwp

    hxt15, hyt15, hzt15 = tail15_unshielded(dmidn, pwr, xc, rn, x, y, z)
    fxt15, fyt15, fzt15 = tail15_shld(x, y, z, bzimf, pwr, xc, rn)

    bx = (hxt15 + fxt15) * t15fact
    by = (hyt15 + fyt15) * t15fact
    bz = (hzt15 + fzt15) * t15fact

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def src_shielded(x, y, z, parameters):
    bzimf = parameters[-3]
    srcscl = parameters[11]
    srceps = parameters[12]

    hxsrc, hysrc, hzsrc = src_unsh(srceps, srcscl, x, y, z)
    fxsrc, fysrc, fzsrc = src_shld(x, y, z, bzimf, srceps, srcscl)

    bx = hxsrc + fxsrc
    by = hysrc + fysrc
    bz = hzsrc + fzsrc

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def prc_shielded(x, y, z, parameters):
    bzimf = parameters[-3]
    prcscl = parameters[13]
    prceps = parameters[14]
    prc_phi = parameters[15]

    cprc = np.cos(prc_phi)
    sprc = np.sin(prc_phi)

    # Call corresponding Python functions (need to be defined elsewhere)
    hxprc1, hyprc1, hzprc1 = prc_unsh_nm(prceps, prcscl, x, y, z)
    fxprc1, fyprc1, fzprc1 = prc_shld_nm(x, y, z, bzimf, prceps, prcscl)

    bnmx = hxprc1 + fxprc1
    bnmy = hyprc1 + fyprc1
    bnmz = hzprc1 + fzprc1

    hxprc2, hyprc2, hzprc2 = prc_unsh_dd(prceps, prcscl, x, y, z)
    fxprc2, fyprc2, fzprc2 = prc_shld_dd(x, y, z, bzimf, prceps, prcscl)

    bddx = hxprc2 + fxprc2
    bddy = hyprc2 + fyprc2
    bddz = hzprc2 + fzprc2

    hxprcs, hyprcs, hzprcs = prcs_unsh(prceps, prcscl, x, y, z)
    fxprcs, fyprcs, fzprcs = prcs_shld(x, y, z, bzimf, prceps, prcscl)

    bsymx = hxprcs + fxprcs
    bsymy = hyprcs + fyprcs
    bsymz = hzprcs + fzprcs

    bx = -bnmx * cprc + bddx * sprc + bsymx
    by = -bnmy * cprc + bddy * sprc + bsymy
    bz = -bnmz * cprc + bddz * sprc + bsymz

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def deform_xz_yz(ps, parameters, x, y, z, b_ext_untitled):
    a = np.array([
        0.6563406936, 0.4596362842, -0.1206495537, 0.3175118510,
        0.3546433781, -0.1987301198, -0.06233108192, 0.2040942759,
        0.02718066023, -0.02798547019, -0.01610526, -0.001069132928,
        0.02242677676, -0.05679826540, -0.002772268386, -0.00258946203,
        0.003512794890, 0.004717943878, 0.5878180276, 0.03796190668,
        -0.1438414960, 0.6721710243, 0.03461738529, -0.7534481962,
        -0.1278768535, 0.4255447972, 0.07119080497
    ])

    rh0 = 8.

    rh = parameters[10]
    bzimf = parameters[-3]

    frh = (rh - rh0) / rh0
    fbz = bzimf / 5.0

    b1 = a[0] + fbz * a[9] + frh * a[18]
    b2 = a[1] + fbz * a[10] + frh * a[19]
    b3 = a[2] + fbz * a[11] + frh * a[20]
    b4 = a[3] + fbz * a[12] + frh * a[21]
    b5 = a[4] + fbz * a[13] + frh * a[22]
    b6 = a[5] + fbz * a[14] + frh * a[23]
    b7 = a[6] + fbz * a[15] + frh * a[24]
    b8 = a[7] + fbz * a[16] + frh * a[25]
    b9 = a[8] + fbz * a[17] + frh * a[26]

    rh2 = rh ** 2
    sps = np.sin(ps)
    cps = np.cos(ps)
    rho2xz = x ** 2 + z ** 2
    rhoxz = np.sqrt(rho2xz)
    rho4rh4 = np.sqrt(np.sqrt(rho2xz ** 2 + rh2 ** 2))

    if x == 0.0 and z == 0.0:
        phi1 = 0.0
        cphi1 = 1.0
        sphi1 = 0.0
    else:
        phi1 = np.arctan2(z, x)
        cphi1 = x / rhoxz
        sphi1 = z / rhoxz

    s = ps * (rho4rh4 - rh)
    dsdrho = ps * rho2xz * rhoxz / rho4rh4 ** 3

    sqt = s + rhoxz * np.sin(ps + phi1)
    dtdrho = 2.0 * sqt * (dsdrho + np.sin(ps + phi1))
    dtdphi = 2.0 * sqt * rhoxz * np.cos(ps + phi1)
    t = sqt ** 2

    f = 1.0 - (b1 + b2 * cphi1 + b3 * (2.0 * cphi1 ** 2 - 1.0)) * t / (rh2 + t)
    dfdrho = -rh2 * dtdrho / (rh2 + t) ** 2 * (b1 + b2 * cphi1 + b3 * (2.0 * cphi1 ** 2 - 1.0))
    dfdphi = -rh2 / (rh2 + t) ** 2 * dtdphi * (b1 + b2 * cphi1 + b3 * (2.0 * cphi1 ** 2 - 1.0)) + (b2 * sphi1 + 4.0 * b3 * sphi1 * cphi1) / (rh2 + t) * t

    br1 = rho4rh4 + rh
    br2 = rho4rh4 ** 2 + rh ** 2
    phis1 = phi1 + f * ps * (1.0 + rho2xz * rhoxz / br1 * cphi1 / br2)

    rhos_over_rhoxz = (1.0 + rhoxz / rh ** 2 * (ps * sphi1 * (b4 + b5 * sphi1 ** 2) + b6 * (ps * sphi1) ** 2)
                       + rhoxz ** 2 / rh ** 3 * (ps * sphi1 * (b7 + b8 * sphi1 ** 2) + b9 * (ps * sphi1) ** 2))

    rhos = rhos_over_rhoxz * rhoxz

    drhosdrho = (1.0 + 2.0 * (rhoxz / rh ** 2) * (ps * sphi1 * (b4 + b5 * sphi1 ** 2) + b6 * (ps * sphi1) ** 2)
                 + 3.0 * (rhoxz / rh) ** 2 / rh * (ps * sphi1 * (b7 + b8 * sphi1 ** 2) + b9 * (ps * sphi1) ** 2))

    drhosdphi_over_rhoxz = (rhoxz / rh ** 2 * ps * (b4 + 3.0 * b5 * sphi1 ** 2 + 2.0 * b6 * ps * sphi1) * cphi1
                            + rhoxz ** 2 / rh ** 3 * ps * (b7 + 3.0 * b8 * sphi1 ** 2 + 2.0 * b9 * ps * sphi1) * cphi1)

    drhosdphi = drhosdphi_over_rhoxz * rhoxz

    dphis1drho = ps * (dfdrho * (1.0 + rho2xz * rhoxz * cphi1 / (br1 * br2))
                       + f * cphi1 * rho2xz * (1.0 / rho4rh4 ** 3 - 1.0 / (br1 * br2)))

    dphis1dphi = (1.0 + ps * (dfdphi * (1.0 + rho2xz * rhoxz * cphi1 / (br1 * br2))
                              - f * rho2xz * rhoxz * sphi1 / (br1 * br2)))

    cphis1 = np.cos(phis1)
    sphis1 = np.sin(phis1)
    xas1 = rhos * cphis1
    zas1 = rhos * sphis1

    rho2yz = y ** 2 + zas1 ** 2
    rhoyz = np.sqrt(rho2yz)
    rho4rh4 = np.sqrt(np.sqrt(rho2yz ** 2 + rh2 ** 2))

    if y == 0.0 and zas1 == 0.0:
        phi2 = 0.0
        cphi2 = 1.0
        sphi2 = 0.0
    else:
        phi2 = np.arctan2(zas1, y)
        cphi2 = y / rhoyz
        sphi2 = zas1 / rhoyz

    g = np.exp(xas1 / (2.0 * rh))  # scale equal to 2*rh (doubled hinging distance)
    dgdx = g / (2.0 * rh)

    f = rho2yz / ((rh + rho4rh4) * (rho4rh4 ** 2 + rh2))
    phis2 = phi2 + f * cphi2 * ps * g * rhoyz
    cphis2 = np.cos(phis2)
    sphis2 = np.sin(phis2)

    dphis2drho = ps * cphi2 * g * rh * f / rho4rh4 ** 3 * (rho4rh4 ** 2 + rh2 + rh * rho4rh4)
    dphis2dphi = 1.0 - sphi2 * ps * g * f * rhoyz
    dphis2dx = (phis2 - phi2) * dgdx / g

    yas2 = rhoyz * cphis2
    zas2 = rhoyz * sphis2

    bx_as2, by_as2, bz_as2 = b_ext_untitled(xas1, yas2, zas2, parameters)

    brho_as2 = by_as2 * cphis2 + bz_as2 * sphis2
    bphi_as2 = -by_as2 * sphis2 + bz_as2 * cphis2

    brho_s2 = brho_as2 * dphis2dphi
    bphi_s2 = bphi_as2 - rhoyz * (bx_as2 * dphis2dx + brho_as2 * dphis2drho)
    bx_s2 = bx_as2 * dphis2dphi
    by_s2 = brho_s2 * cphi2 - bphi_s2 * sphi2
    bz_s2 = brho_s2 * sphi2 + bphi_s2 * cphi2

    brho_as1 = bx_s2 * cphis1 + bz_s2 * sphis1
    bphi_as1 = -bx_s2 * sphis1 + bz_s2 * cphis1

    brho_s1 = (brho_as1 * rhos_over_rhoxz * dphis1dphi -
               bphi_as1 * drhosdphi_over_rhoxz)

    bphi_s1 = (-brho_as1 * rhos * dphis1drho +
               bphi_as1 * drhosdrho)

    by = (by_s2 * rhos_over_rhoxz *
          (drhosdrho * dphis1dphi - drhosdphi * dphis1drho))

    bx = brho_s1 * cphi1 - bphi_s1 * sphi1
    bz = brho_s1 * sphi1 + bphi_s1 * cphi1

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def dipole_shield(x, y, z, psi, pdpm, bzimf):
    p = np.array([0.01447178814, 0.08413473832, 0.1991022951, 0.2550624546])
    q = np.array([0.04257168124, 0.1103920722, 0.2573823975, 0.2625976694])
    a = np.array([
        63.86081318, -0.3885695056, -2.849842473, -60.28997524, 3.233295175, 2.871808009,
        -73.62746378, -8.245643028, 6.595835800, -6.951805745, 2.800347308, -1.356635839,
        -5.143464615, 6.408399581, -6.067725736, 30.67977107, 5.000396216, -1.055729325,
        3.419893572, -3.075383297, 2.659892266, -19.75394468, -4.566011794, 2.951809155,
        -109.7271663, -34.43568959, 27.41946353, -95.07105895, 9.843457076, 1.719957805,
        -22.79541332, 12.99857748, -11.40354272, 44.02049830, 0.6871863029, 2.199806700,
        6.757295444, -1.408254083, 0.5197439507, -27.60415416, -10.25059155, 4.134026820,
        0.07404641498, 2.238492418, -1.864478684, 16.84429987, 6.769569345, -3.726813564,
        162.5809875, 67.22623843, -58.30248283, -27.72898928, -169.2137151, 42.14614616,
        -37.34007191, -16.98302872, 16.18609972, 19.73195259, 44.50393218, -18.41892067,
        11.97092245, 8.064302756, -9.609617953, -13.33814648, -12.97520543, 13.68442596,
        -3.468923614, -4.018150510, 4.689195165, 2.036783832, 4.080251737, -5.416868377,
        -80.84117540, -45.53695917, 21.29724170, -196.1425005, 139.1928444, 10.81721683,
        21.07960678, 12.10130166, -7.425059208, 37.59242095, -37.11111910, 3.418232069,
        -5.696353206, -4.803016086, 4.970713066, -0.9094040713, 10.96145505, -7.336800977,
        1.899094764, 2.032232484, -2.243069677, 0.01098379146, -3.720170927, 3.270198923,
        6.453873059, -0.3456294745, 0.06299853220, -6.121609605, 0.3728488180, 0.1615893793,
        -48.10449335, -20.31468938, 21.11181314, 2.555708375, 25.01753057, -36.79671388,
        -105.1198326, -9.203192329, 12.26438789, 27.13684161, 14.83319028, -22.52796872,
        57.12677447, 42.57944976, -57.08889030, 54.52422848, -23.45756314, 21.76700933,
        -52.40139083, -39.73980409, 53.82491643, -50.71497018, 23.25099793, -22.20835644,
        -57.87812254, -5.289278367, 6.664141243, 6.258399887, 11.51824518, -15.16654009,
        -1.941465351, 3.421124393, -2.587475652, 16.70299906, 0.9129739383, -8.260190639,
        -38.71315542, -15.95894571, 30.52336081, -98.21679787, 48.74035160, -23.63976948,
        39.87167804, 14.57342294, -28.90843517, 92.25160373, -44.66671904, 20.43199614,
        1.153643473, -6.738510233, 7.189828136, -22.19675170, 65.32693194, -54.92558715,
        -8.414249373, 29.34844552, -27.20567962, 32.30112038, -79.30209098, 60.88466980,
        449.2928512, 110.7275433, -234.6865986, 73.64104049, -2.021078006, -14.24961044,
        -423.6136930, -107.7302335, 225.4358329, -84.25699978, 5.750942072, 19.30387318,
        2.179232843, 7.744352121, -9.612057958, 13.63358628, -68.31902318, 62.40732796,
        5.641664969, -30.12199545, 30.13377902, -19.39562341, 87.13214102, -76.43904826,
        -407.1903205, -98.05581795, 208.9313367, -89.98673432, -41.31483346, 70.22247312,
        384.2137860, 95.43979917, -200.8837009, 97.66803457, 34.50569490, -70.35473096,
        -0.03043796699, 0.7762568873, -0.8998460979, -0.7098931980, -1.289923999, 1.986337283,
        0.01447178814, 0.08413473832, 0.1991022951, 0.2550624546, 0.04257168124, 0.1103920722,
        0.2573823975, 0.2625976694, 0.007183038459, 0.007537449434, 1.425884491, 1.124702200
    ])

    fact1, fact2 = 0.007183038459, 0.007537449434
    xd1, xd2 = 28.51768982, 22.494044

    fx = 0.0
    fy = 0.0
    fz = 0.0

    ps2 = psi ** 2
    fpdpm = (pdpm / 2.01) ** 0.194  # Power index from Lin et al. [2010]
    fpdpm3 = fpdpm ** 3
    xx = x * fpdpm
    yy = y * fpdpm
    zz = z * fpdpm

    fbz = bzimf / 5.0
    fbz21 = np.abs(fbz) / np.sqrt(1.0 + fact1 * fbz ** 2)
    fbz22 = np.abs(fbz) / np.sqrt(1.0 + fact2 * fbz ** 2)

    l = 0  # Adjusted for 0-based indexing (Fortran L=1 -> Python l=0)

    # First loop: Perpendicular components with P scales
    for i in range(4):
        pi = p[i]
        cypi = np.cos(yy * pi)
        sypi = np.sin(yy * pi)
        for k in range(4):
            pk = p[k]
            szpk = np.sin(zz * pk)
            czpk = np.cos(zz * pk)
            sqpp = np.sqrt(pi ** 2 + pk ** 2)
            epp = np.exp(xx * sqpp)

            hx0 = -sqpp * epp * cypi * szpk
            hy0 = epp * sypi * szpk * pi
            hz0 = -epp * cypi * czpk * pk

            hx1 = hx0 * fbz
            hy1 = hy0 * fbz
            hz1 = hz0 * fbz

            hx2 = hx0 * fbz21
            hy2 = hy0 * fbz21
            hz2 = hz0 * fbz21

            al = a[l]
            al1 = a[l + 1]
            al2 = a[l + 2]
            al3 = a[l + 3]
            al4 = a[l + 4]
            al5 = a[l + 5]

            fx += al * hx0 + al1 * hx1 + al2 * hx2 + (al3 * hx0 + al4 * hx1 + al5 * hx2) * ps2
            fy += al * hy0 + al1 * hy1 + al2 * hy2 + (al3 * hy0 + al4 * hy1 + al5 * hy2) * ps2
            fz += al * hz0 + al1 * hz1 + al2 * hz2 + (al3 * hz0 + al4 * hz1 + al5 * hz2) * ps2

            l += 6

    # Placeholder for DIPOLE subroutine (perpendicular)
    dperpx, dperpy, dperpz = dipole(0.0, xx - xd1, yy, zz)  # You need to define this

    fx += (a[l] + a[l + 1] * fbz + a[l + 2] * fbz21 + a[l + 3] * ps2 +
           a[l + 4] * ps2 * fbz + a[l + 5] * ps2 * fbz21) * dperpx
    fy += (a[l] + a[l + 1] * fbz + a[l + 2] * fbz21 + a[l + 3] * ps2 +
           a[l + 4] * ps2 * fbz + a[l + 5] * ps2 * fbz21) * dperpy
    fz += (a[l] + a[l + 1] * fbz + a[l + 2] * fbz21 + a[l + 3] * ps2 +
           a[l + 4] * ps2 * fbz + a[l + 5] * ps2 * fbz21) * dperpz

    l += 6

    # Second loop: Parallel components with Q scales
    for i in range(4):
        qi = q[i]
        cyqi = np.cos(yy * qi)
        syqi = np.sin(yy * qi)
        for k in range(4):
            qk = q[k]
            szqk = np.sin(zz * qk)
            czqk = np.cos(zz * qk)
            sqqq = np.sqrt(qi ** 2 + qk ** 2)
            eqq = np.exp(xx * sqqq)

            hx0 = -sqqq * eqq * cyqi * czqk * psi
            hy0 = eqq * syqi * czqk * qi * psi
            hz0 = eqq * cyqi * szqk * qk * psi

            hx1 = hx0 * fbz
            hy1 = hy0 * fbz
            hz1 = hz0 * fbz

            hx2 = hx0 * fbz22
            hy2 = hy0 * fbz22
            hz2 = hz0 * fbz22

            al = a[l]
            al1 = a[l + 1]
            al2 = a[l + 2]
            al3 = a[l + 3]
            al4 = a[l + 4]
            al5 = a[l + 5]

            fx += al * hx0 + al1 * hx1 + al2 * hx2 + (al3 * hx0 + al4 * hx1 + al5 * hx2) * ps2
            fy += al * hy0 + al1 * hy1 + al2 * hy2 + (al3 * hy0 + al4 * hy1 + al5 * hy2) * ps2
            fz += al * hz0 + al1 * hz1 + al2 * hz2 + (al3 * hz0 + al4 * hz1 + al5 * hz2) * ps2

            l += 6

    dparax, dparay, dparaz = dipole(1.570796327, xx - xd2, yy, zz)  # You need to define this

    fx += (a[l] + a[l + 1] * fbz + a[l + 2] * fbz21 + a[l + 3] * ps2 +
           a[l + 4] * ps2 * fbz + a[l + 5] * ps2 * fbz21) * dparax * psi
    fy += (a[l] + a[l + 1] * fbz + a[l + 2] * fbz21 + a[l + 3] * ps2 +
           a[l + 4] * ps2 * fbz + a[l + 5] * ps2 * fbz21) * dparay * psi
    fz += (a[l] + a[l + 1] * fbz + a[l + 2] * fbz21 + a[l + 3] * ps2 +
           a[l + 4] * ps2 * fbz + a[l + 5] * ps2 * fbz21) * dparaz * psi

    # Final scaling
    bx = fx * fpdpm3
    by = fy * fpdpm3
    bz = fz * fpdpm3

    return bx, by, bz


@jit(fastmath=True, nopython=True)
def tail15_unshielded(dmidn, pw, xc, rn, x, y, z):
    a0, a1, a2, a3, a4 = -0.3068451808, -0.006047641772, 0.05689361754, 0.184348423, 0.07165072182
    b0, b1, b2, b3, b4 = 0.2499990832, 0.3119753229, 0.2950446164, 0.1672154523, 0.02133127189
    pi = np.pi
    factor = 400.0  # Normalization factor

    # Parameters
    dd = 7.0  # Flankward flaring parameter
    drh = 5.0  # Regularization parameter
    step = 0.5  # Integration step size
    ri = rn + xc  # Starting radius

    # Initialize field components
    bx, by, bz = 0.0, 0.0, 0.0

    # Calculate rho and D parameters
    rho = np.sqrt((xc - x) ** 2 + y ** 2)
    d = dmidn + dd * (rho - xc + x) / 2.0 * rho / (rho ** 2 + drh ** 2)
    dddrho = dd * drh ** 2 * (rho - xc + x) / (rho ** 2 + drh ** 2) ** 2

    # Main integration loop
    for i in range(1, 301):
        r = ri + (i - 1) * step
        amp = -(rn / (r - xc)) ** pw

        if rho <= 1e-8:
            # Very close to z-axis case
            dbx, dby = 0.0, 0.0
            dbz = (pi / 4.0) * r / np.sqrt(r ** 2 + z ** 2 + d ** 2) ** 3 * factor
        else:
            # Handle small rho case with scaling
            if rho <= 1e-3:
                scale = 1e-3 / rho
            else:
                scale = 1.0

            x1 = (x - xc) * scale
            y1 = y * scale
            rh1 = rho * scale

            # Main field calculations
            sqb = (r + rh1) ** 2 + z ** 2 + d ** 2
            p = 1.0 - 4.0 * r * rh1 / sqb

            # Calculate F and its derivative FS
            f = (a0 + p * (a1 + p * (a2 + p * (a3 + p * a4))) -
                 np.log(p) * (b0 + p * (b1 + p * (b2 + p * (b3 + p * b4)))))

            fs = (a1 + p * (2.0 * a2 + p * (3.0 * a3 + p * 4.0 * a4)) -
                  (b0 + p * (b1 + p * (b2 + p * (b3 + p * b4)))) / p -
                  np.log(p) * (b1 + p * (2.0 * b2 + p * (3.0 * b3 + p * 4.0 * b4))))

            # Calculate field components
            brho = z / np.sqrt(sqb) ** 3 * (f - fs / sqb * 8.0 * r * rh1) * factor
            dbz = (f / rh1 - (f * (r + rh1 + d * dddrho) +
                              4.0 * r / sqb * fs * (
                                          r ** 2 - rh1 ** 2 + z ** 2 + d ** 2 - 2.0 * d * rh1 * dddrho)) / sqb)
            dbz = dbz / np.sqrt(sqb) * factor

            dbx = brho / rh1 * x1 / scale
            dby = brho / rh1 * y1 / scale

        # Accumulate field components
        bx += dbx * amp
        by += dby * amp
        bz += dbz * amp

    return bx, by, bz


@jit(fastmath=True, nopython=True)
def tail15_shld(x, y, z, bzimf, pw, xc, rn):
    nlin = 509
    rmax0 = 11.0

    a = np.array([
        -77.299735, 144.91631, -91.057277, -2971.3349,
        113.54416, 35.663838, 1861.5598, -12.396176,
        304.55727, -211.56189, 103.51096, -71.726085,
        -1489.4013, -65.883343, 66.252460, 107.40580,
        -377.64247, 231.18301, 107.20919, -549.38825,
        -355.03208, 1557.3685, 86.964365, -638.09932,
        -293.30601, 669.46881, 563.20128, 289.55635,
        -38.270496, -137.76836, 40.322425, 15.783851,
        -307.98049, 268.63108, 61.467583, -219.97292,
        -51.607951, 61.989688, -247.68114, 53.642657,
        197.28704, -361.04334, 1960.1476, 23.278720,
        202.74748, -1188.9041, 839.31271, 533.15725,
        488.15395, -2.9048013, 108.98846, 48.231822,
        -27.318735, 153.38323, -645.00110, 4.6456912,
        138.58784, -17.964537, -454.25003, 47.790107,
        -11.822084, -607.91138, -206.24438, 1676.2319,
        86.375501, -254.54485, -568.17535, 943.27563,
        669.36923, 851.69758, 27.384575, -66.444513,
        -79.140068, 67.127566, -204.55387, -596.23477,
        -76.118178, -212.47855, 53.662951, -240.62353,
        -46.489539, 39.520590, -370.47347, -149.39785,
        1897.7764, 212.09286, -262.34758, -762.37973,
        1061.1355, 724.64006, 465.81606, -15.267786,
        26.024148, -19.186665, 183.45034, -143.24679,
        -608.01903, -35.883648, -169.40933, 35.113410,
        -426.53431, -149.10499, 62.658793, -621.87964,
        -6.3928884, 2373.7081, 92.982535, -82.156689,
        -964.36573, 1121.9716, 740.33337, 920.55078,
        -0.39154359, -8.8822768, -45.841782, 80.189853,
        -97.279282, -430.38857, 16.752319, -21.800884,
        78.835637, -222.41397, -31.910808, 55.998459,
        -316.86458, -97.596877, -988.66329, -88.531687,
        12.813123, -159.16152, -347.29918, -247.20316,
        537.19663, 12.964520, 53.720638, -128.39067,
        -41.852226, 39.415405, -50.614435, -37.201993,
        42.565835, 19.422873, 208.19497, 102.13846,
        7.4315672, 314.41313, -151.93639, -449.31859,
        -8.6423762, -206.75967, 915.69882, -345.00648,
        -161.36385, -625.89136, -27.208018, -84.782112,
        56.368211, -19.522568, -44.502500, 780.89478,
        61.210380, -47.889991, -63.278656, 311.31771,
        -127.31346, -4.6156762, 519.29283, -75.570647,
        -611.95787, -45.383652, 385.05162, -1.1786334,
        -481.21512, -374.35092, -394.15952, 3.2438765,
        128.94143, 42.123911, -58.164040, 254.96543,
        31.665308, 14.336831, 213.24856, -24.776273,
        -41.187272, 126.78179, -45.849804, -42.600682,
        48.287467, -1069.9539, -56.006882, -75.600748,
        657.24072, -503.53005, -319.30797, -106.94351,
        28.266515, -73.568742, -50.742379, -35.911130,
        -90.133229, 234.11186, -45.154205, -104.50833,
        14.506305, 255.33476, 32.846310, -12.512339,
        332.50122, 32.877102, -975.84190, -5.5238290,
        47.943987, 390.88428, -460.93763, -323.81457,
        -590.62076, -11.699011, 34.413499, 45.268957,
        14.724945, 66.822507, 189.87360, 1.8328859,
        17.654656, -38.527452, 28.557558, -31.766343,
        -18.139067, 34.453056, 141.25647, -781.49228,
        -114.21063, 152.99248, 306.16453, -477.96714,
        -349.12922, -237.30717, 7.4733664, -8.6682248,
        20.337126, -78.059104, 65.729272, 293.58294,
        24.285912, 87.852207, -9.9784521, 197.62640,
        59.232202, -31.619550, 270.67970, 120.03945,
        -1239.8713, -211.53366, 106.04331, 1205.9156,
        -425.59465, -304.72494, -355.80970, 28.206260,
        -137.88278, -19.990168, -99.741167, 108.04172,
        509.83826, -23.154102, 16.159284, -36.090955,
        334.71036, 13.480783, -55.417668, 474.51257,
        33.748552, -1512.4720, -44.358183, 130.23970,
        1306.5833, -816.83076, -537.22929, -1116.6542,
        -15.966669, -38.815477, 73.628171, -4.7370903,
        133.29417, 765.20960, 43.546208, 65.357403,
        -51.747880, 277.27141, -35.066069, -37.933569,
        413.33117, 223.83057, -2069.7591, -59.779377,
        428.42883, 578.64921, -1190.3341, -866.73224,
        -786.80860, 7.2461799, 125.14143, 19.497124,
        -47.604326, 250.38011, 352.66425, 15.613848,
        216.93456, -7.1068543, 190.15406, 160.63733,
        -50.982616, 254.95280, 279.63302, -2566.1350,
        -152.33939, -3.2813966, 1326.5931, -1299.5151,
        -860.07391, -567.96050, 23.705746, -101.24105,
        -35.841136, -110.16262, -48.331271, 783.32137,
        -10.824806, -34.363522, -3.9167162, 596.91441,
        74.994836, -36.970159, 823.20477, 126.67412,
        -2550.4585, -170.21107, 309.16020, 963.63527,
        -1317.5256, -915.62501, -1171.7935, -11.985488,
        43.724305, 96.827542, -123.31275, 251.41280,
        674.34043, 43.330829, 199.00045, -93.467195,
        292.86929, 65.823833, -73.398503, 436.88285,
        157.29372, -2735.4453, -203.65228, 233.03764,
        1106.0569, -1421.3273, -974.75338, -955.46520,
        11.490972, -9.1189979, 53.107594, -153.59757,
        138.94824, 659.75450, 11.020499, 107.62402,
        -69.670796, 406.59534, 97.672706, -80.266083,
        563.37599, 228.72899, 656.64010, -141.54978,
        -15.915640, 310.96192, 650.62919, 401.34201,
        -39.770055, 28.692408, -176.36590, 62.487646,
        -23.566163, -6.1348345, -222.59809, -40.661225,
        -103.59647, -27.692213, -183.13756, -92.054085,
        -39.716945, -327.60975, 69.041307, 1219.1920,
        104.18408, -5.4084372, -514.47276, 747.75236,
        472.83283, -8.3467961, -12.395795, 11.560202,
        64.281785, 134.42826, -25.137459, -533.03678,
        -5.8188090, -57.088459, 18.547149, -441.77460,
        -96.825450, 17.240020, -691.76842, 264.74377,
        1302.0372, 133.99073, -5.1074071, -1107.9965,
        656.84115, 392.66839, 706.84211, 0.57802506,
        92.844715, -50.025827, 104.68769, -81.383713,
        -652.03344, -9.2735507, -10.714266, 83.517954,
        -314.51945, 20.674744, 45.629849, -498.13106,
        160.94771, 1396.0982, 18.228963, -371.93074,
        -364.46842, 754.46872, 557.22932, 1019.4198,
        5.1169307, -145.74957, -90.174362, -0.41299552,
        -296.03289, -54.513013, 2.8270670, -167.04331,
        72.859665, 140.35115, -50.291869, 59.506179,
        196.91365, -221.37274, 1681.8806, 20.608970,
        131.63976, -900.12911, 843.00952, 568.65257,
        377.48664, -25.890133, 62.748246, 48.134962,
        -10.895728, 160.88251, -322.97381, 57.473476,
        175.01825, -25.888580, -316.42995, -2.4727534,
        15.867690, -370.96957, -336.06817, 1227.5772,
        210.35213, -316.11337, -423.05764, 747.95374,
        561.55358, 496.53624, -3.1281332, -1.3305618,
        -65.152448, 137.61672, -177.78877, -520.99581,
        -61.182452, -210.49264, 32.996368, -289.57438,
        -87.035130, 55.447545, -410.27431, -179.23546,
        57.271902, -3.1614126, -4.2402151, -102.00960,
        -6.4069034, 4.1626399, 0.0011085764, 0.0024442371,
        -1.4296991, -1.5673126, 0.00025464023, -0.000010305815,
        0.00000068969354, -0.00033691538, 0.00091769431, 6.7307586,
        -2.2051366, 0.0000038801141, 0.0000050020353, 0.000026978491,
        -0.000087027859, -0.0063041822, -0.0027773576, -0.0013934338,
        -0.16410856
    ])

    bzimf_low = -10.0
    bzimf_high = 10.0
    pw_low = -0.6
    pw_high = 0.6
    xc_low = -2.0
    xc_high = 5.0
    rn_low = 4.0
    rn_high = 10.0

    fbz = (bzimf * 2.0 - bzimf_high - bzimf_low) / (bzimf_high - bzimf_low)
    fbz21 = np.abs(fbz)
    fpw = (pw * 2.0 - pw_high - pw_low) / (pw_high - pw_low)
    fxc = (xc * 2.0 - xc_high - xc_low) / (xc_high - xc_low)
    frn = (rn * 2.0 - rn_high - rn_low) / (rn_high - rn_low)

    a0 = a[nlin + 1] + a[nlin + 6] * fbz + a[nlin + 7] * fbz21 + a[nlin + 13] * frn + a[nlin + 14] * fxc + a[
        nlin + 15] * fpw
    b0 = a[nlin + 2] + a[nlin + 8] * fbz + a[nlin + 9] * fbz21 + a[nlin + 16] * frn + a[nlin + 17] * fxc + a[
        nlin + 18] * fpw

    rmax_thin = rmax0 * (a[nlin + 10] + a[nlin + 4] * fbz + a[nlin + 5] * fbz21 + a[nlin + 19] * frn)
    shift1 = a[nlin + 11] * 20.0
    shift2 = a[nlin + 12] * 20.0

    fx = 0.0
    fy = 0.0
    fz = 0.0

    ind = -21

    for i in range(1, 5):
        for k in range(1, 7):
            a_i = a0 * i
            b_k = b0 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)
            bx = xk * ex * cay * sbz
            by = -a_i * ex * say * sbz
            bz = b_k * ex * cay * cbz

            ind += 21

            ai = (a[ind] + a[ind + 1] * fbz + a[ind + 2] * fxc + a[ind + 3] * fpw + a[ind + 4] * fbz21 +
                  a[ind + 5] * fxc ** 2 + a[ind + 6] * fpw ** 2 + a[ind + 7] * fbz * fxc + a[ind + 8] * fxc * fpw +
                  a[ind + 9] * fbz * fpw + a[ind + 10] * fbz * fbz21 + a[ind + 11] * fxc ** 3 + a[ind + 12] * fpw ** 3 +
                  a[ind + 13] * fbz * fxc * fpw + a[ind + 14] * fbz21 * fxc + a[ind + 15] * fbz * fxc ** 2 +
                  a[ind + 16] * fxc ** 2 * fpw + a[ind + 17] * fxc * fpw ** 2 + a[ind + 18] * fbz * fpw ** 2 +
                  a[ind + 19] * fbz21 * fpw + a[ind + 20] * frn)

            fx += ai * bx
            fy += ai * by
            fz += ai * bz

    ind += 21
    bxn, byn, bzn = t89_2_disk_thin(x - shift1, y, z - 50.0, rmax_thin)
    bxs, bys, bzs = t89_2_disk_thin(x - shift1, y, z + 50.0, rmax_thin)

    ai = a[ind] + a[ind + 1] * fbz + a[ind + 2] * fbz21
    fx += ai * (bxn + bxs)
    fy += ai * (byn + bys)
    fz += ai * (bzn + bzs)

    ind += 3
    bxn, byn, bzn = t89_2_disk_thin(x - shift2, y, z - 70.0, rmax_thin)
    bxs, bys, bzs = t89_2_disk_thin(x - shift2, y, z + 70.0, rmax_thin)

    ai = a[ind] + a[ind + 1] * fbz + a[ind + 2] * fbz21
    fx += ai * (bxn + bxs)
    fy += ai * (byn + bys)
    fz += ai * (bzn + bzs)

    return fx, fy, fz

@jit(fastmath=True, nopython=True)
def t89_2_disk_thin(x, y, z, rmax):
    r0 = rmax*1.41421356
    r02 = - r0**2
    rho2 = x**2 + y**2
    zr = np.abs(z) + r0
    s = np.sqrt(rho2 + zr**2)
    szr = s + zr
    sigz = np.sign(z)
    fxy = sigz / s ** 3 * r02
    bx = fxy * x
    by = fxy * y
    bz = -np.abs(fxy) / szr * zr * (2.0 * s - rho2 / szr)

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def src_unsh(eps, scale, x, y, z):
    dt0 = 0.07
    dr0 = 0.17

    xsc = x * scale
    ysc = y * scale
    zsc = z * scale

    # Convert Cartesian to spherical
    rho_sc = np.sqrt(ysc ** 2 + zsc ** 2)
    rsc = np.sqrt(xsc ** 2 + ysc ** 2 + zsc ** 2)
    t = np.arctan2(rho_sc, xsc)
    ts = t - dt0 * rsc ** eps * rho_sc / rsc  # Deformed theta_s = theta - dt0 * r^eps * sin(theta)
    dts_dr = -dt0 * eps * rsc ** (eps - 1) * rho_sc / rsc  # Derivative dtheta_s/dr
    dts_dt = 1.0 - dt0 * rsc ** eps * xsc / rsc  # Derivative dtheta_s/dtheta

    r_prime = rsc * (1.0 - dr0 * np.sin(t / 2.0) ** 2)
    dr_prime_dr = 1.0 - dr0 * np.sin(t / 2.0) ** 2
    dr_prime_dt = -dr0 * rsc * 0.5 * np.sin(t)

    if rho_sc > 1e-5:
        cp = ysc / rho_sc
        sp = zsc / rho_sc
        stsst = np.sin(ts) / np.sin(t)
    else:
        cp = 1.0
        sp = 0.0
        stsst = dts_dt

    x_s = r_prime * np.cos(ts)
    y_s = r_prime * np.sin(ts) * cp
    z_s = r_prime * np.sin(ts) * sp

    bxas, byas, bzas = src_axisymmetric(x_s, y_s, z_s)

    rho2_s = y_s ** 2 + z_s ** 2
    r_s = np.sqrt(rho2_s + x_s ** 2)
    rho_s = np.sqrt(rho2_s)

    ct_s = x_s / r_s
    st_s = rho_s / r_s

    bras = (x_s * bxas + y_s * byas + z_s * bzas) / r_s  # Convert from Cartesian to spherical
    bthetaas = (byas * cp + bzas * sp) * ct_s - bxas * st_s
    bphias = bzas * cp - byas * sp

    br_s = stsst * dts_dt * (r_prime / rsc) ** 2 * bras - r_prime / rsc ** 2 * stsst * dr_prime_dt * bthetaas
    btheta_s = -r_prime ** 2 / rsc * stsst * dts_dr * bras + stsst * (r_prime / rsc) * dr_prime_dr * bthetaas
    bph_s = (r_prime / rsc) * (dr_prime_dr * dts_dt - dr_prime_dt * dts_dr) * bphias

    bxsc = br_s * np.cos(t) - btheta_s * np.sin(t)  # Convert from spherical to Cartesian
    bmerid = br_s * np.sin(t) + btheta_s * np.cos(t)
    bysc = bmerid * cp - bph_s * sp
    bzsc = bmerid * sp + bph_s * cp

    return bxsc, bysc, bzsc

@jit(fastmath=True, nopython=True)
def src_axisymmetric(x, y, z):
    f1, f2 = -42.6031, 2.16189
    dr, r1, d1 = 3.76802, 5.70915, 1.98537
    r2, d2 = 2.48923, 0.607298
    al = 0.931890
    a1, a2, a3, a4, a5, a6, a7 = 0.0520056, -0.510551, -0.570999, 0.236973, 0.166199, 0.133038, 0.0420191
    b1, b2, b3, b4, b5, b6, b7 = 1.21158, -0.241321, -1.51826, -0.539575, 0.144579, 0.257290, 0.134827

    rho = np.sqrt(x ** 2 + y ** 2)
    if rho < 1e-9:
        rho = 1e-9  # Prevent division by zero

    rr2 = ((rho - r1) ** 2 + z ** 2) / dr ** 2
    dex = np.exp(-rr2)

    argr = al * (rho - r1) / dr
    argz = al * z / dr

    sr = np.sin(argr)
    cr = np.cos(argr)
    s2r = 2 * sr * cr
    c2r = cr ** 2 - sr ** 2
    sz = np.sin(argz)
    cz = np.cos(argz)
    s2z = 2 * sz * cz
    c2z = cz ** 2 - sz ** 2

    sumrho = a1 + a2 * sr + a3 * cr + a4 * cz + a5 * s2r + a6 * c2r + a7 * c2z
    brackrho = 1.0 + dex * sumrho
    rho_ast = rho * brackrho

    sumz = b1 + b2 * sr + b3 * cr + b4 * cz + b5 * s2r + b6 * c2r + b7 * c2z
    brackz = 1.0 + dex * sumz
    z_ast = z * brackz

    drhoast_drho = (brackrho - rho * dex * sumrho * 2 / dr ** 2 * (rho - r1) +
                    rho * dex * (a2 * cr - a3 * sr + 2 * a5 * c2r - 2 * a6 * s2r) * al / dr)

    drhoast_dz = (-rho * dex * 2 * z / dr ** 2 * sumrho -
                  rho * dex * (a4 * np.sin(argz) + 2 * a7 * s2z) * al / dr)

    dzast_drho = (-z * 2 * (rho - r1) / dr ** 2 * dex * sumz +
                  z * dex * (b2 * cr - b3 * sr + 2 * b5 * c2r - 2 * b6 * s2r) * al / dr)

    dzast_dz = (brackz - z * dex * 2 * z / dr ** 2 * sumz -
                z * dex * (b4 * np.sin(argz) + 2 * b7 * s2z) * al / dr)

    x_ast = rho_ast * x / rho
    y_ast = rho_ast * y / rho

    bx_ast, by_ast, bz_ast = spread_loop_b(r1, d1, x_ast, y_ast, z_ast)
    brho_ast = (bx_ast * x_ast + by_ast * y_ast) / rho_ast

    brho_s = rho_ast / rho * (dzast_dz * brho_ast - drhoast_dz * bz_ast)
    bphi_s = 0.0

    bx1 = brho_s * x / rho
    by1 = brho_s * y / rho
    bz1 = rho_ast / rho * (-dzast_drho * brho_ast + drhoast_drho * bz_ast)

    bx2, by2, bz2 = spread_loop_b(r2, d2, x, y, z)

    bx = f1 * bx1 + f2 * bx2
    by = f1 * by1 + f2 * by2
    bz = f1 * bz1 + f2 * bz2

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def spread_loop_b(r, d, x, y, z):
    a0, a1, a2, a3, a4 = -0.3068451808, -0.006047641772, 0.05689361754, 0.184348423, 0.07165072182
    b0, b1, b2, b3, b4 = 0.2499990832, 0.3119753229, 0.2950446164, 0.1672154523, 0.02133127189
    factor = 34.0  # Adjusted for Bz = +1 nT at origin

    rho = np.sqrt(x ** 2 + y ** 2)

    # Handle very close to z-axis case
    if rho <= 1e-8:
        bx = 0.0
        by = 0.0
        bz = (np.pi / 4.0) * r / np.sqrt(r ** 2 + z ** 2 + d ** 2) ** 3 * factor
        return bx, by, bz

    # Handle small rho case with scaling
    if rho <= 1e-3:
        scale = 1e-3 / rho
    else:
        scale = 1.0

    x1 = x * scale
    y1 = y * scale
    rh1 = rho * scale

    # Main field calculations
    sqb = (r + rh1) ** 2 + z ** 2 + d ** 2
    p = 1.0 - 4.0 * r * rh1 / sqb

    # Calculate F and its derivative FS
    f = (a0 + p * (a1 + p * (a2 + p * (a3 + p * a4))) -
         np.log(p) * (b0 + p * (b1 + p * (b2 + p * (b3 + p * b4)))))

    fs = (a1 + p * (2.0 * a2 + p * (3.0 * a3 + p * 4.0 * a4)) -
          (b0 + p * (b1 + p * (b2 + p * (b3 + p * b4)))) / p -
          np.log(p) * (b1 + p * (2.0 * b2 + p * (3.0 * b3 + p * 4.0 * b4))))

    # Calculate field components
    brho = z / np.sqrt(sqb) ** 3 * (f - fs / sqb * 8.0 * r * rh1) * factor
    bz = (f / rh1 - (f * (r + rh1) + 4.0 * r / sqb * fs * (r ** 2 - rh1 ** 2 + z ** 2 + d ** 2)) / sqb
          / np.sqrt(sqb) * factor)

    # Scale back if needed
    bx = brho / rh1 * x1 / scale
    by = brho / rh1 * y1 / scale

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def src_shld(x, y, z, bzimf, eps, scale):
    nlin = 400

    a = np.array([
        4.0567599, -3.2150442, -11.606128, 4.0569091, 0.88345389, 16.920233, 0.43084505, 2.1011254,
        -1.6231375, 1.0988260, 0.085336119, 2.9336396, -0.092975675, -1.0484057, 1.0059430, 3.0636431,
        -2.7410025, -0.39395174, 1.4484220, -1.8025719, 10.972415, 5.8446122, 4.5560792, -4.1708999,
        -7.3121544, -23.631832, -1.5022734, -7.4707736, 8.4421859, -0.84427471, 1.2802786, -17.772983,
        -1.8669809, 2.9385014, 3.5599169, -5.6309138, 3.1069949, 1.4914055, -1.6342478, -0.34031158,
        -11.705016, -3.5666562, 4.5062122, 2.7603249, 10.828069, 21.902930, 0.65213025, 6.1327050,
        -11.520702, 0.59635250, -0.73763942, 8.6550318, 0.42447093, -1.7995500, -2.2679181, -0.78533404,
        3.7468857, 1.0407885, 0.63357353, -2.6649078, 8.1375129, -0.68600378, -5.3122802, -3.7207562,
        -7.0787620, -6.5418816, 0.061388770, 5.8818968, 7.3533205, 0.019014250, 3.0419410, 0.48937089,
        2.9654333, -0.094256417, -2.6949448, 0.14858587, -7.5179795, -2.1637333, 1.0933315, 5.4998983,
        -2.2107347, 1.4116049, 0.86824461, 1.8276170, 0.63904086, -1.6169895, -0.29275043, -6.0782656,
        -1.8578759, -0.51020452, -3.4419646, 1.4927971, -2.1392062, 0.80015595, 2.6172124, 2.5821779,
        3.3779296, 0.46641773, -1.0946874, -2.5011705, -14.756312, 11.934882, 4.7870385, -11.681704,
        10.304370, 11.709727, -1.9771252, 7.8470299, -5.4399217, -0.66877010, 0.23515478, 25.329813,
        -0.44374727, -0.69327899, 7.9093064, 10.811746, 1.2980183, 1.0929477, -3.0743639, 2.8471951,
        3.2502413, -15.654648, -11.240328, 7.2597118, -3.9779115, -3.1840833, 0.78406032, 2.3224440,
        -2.4925237, 1.4939157, -5.5929478, -3.6208797, 6.2955249, -1.6672707, -17.425847, -6.1414547,
        -5.4550079, -3.7407740, 2.1844746, 5.8570740, 9.0048780, 9.4118751, -2.4111646, -3.4503672,
        -9.0898024, -6.1527829, 2.2164474, -9.5623747, 13.718593, -3.0919839, 2.3986870, 2.7803577,
        -4.7014664, -0.40013649, 11.137541, 9.2950202, -3.3880186, -0.51536182, 0.26081261, -0.70735412,
        -9.4535407, 0.16020783, 8.2856224, 5.5799290, 8.4563940, -2.5641790, -2.8087744, -8.8316198,
        -12.041722, 2.0510702, -4.0342184, -7.1767567, -3.6114171, 1.8897827, 3.3653286, -0.96926041,
        10.494299, 2.9840327, -2.9551398, -8.4438220, 3.0095613, -2.6626326, -1.1870524, -3.2795824,
        -0.14309899, 6.5783854, 1.3267524, 10.883429, 3.4782674, 0.31784490, 6.0313396, -2.4857622,
        3.7758980, -1.8269256, -5.3219269, -5.7333992, -5.3321527, -0.45541208, 2.2989822, 4.7053565,
        27.206616, -5.4373795, 14.985909, 10.568373, -15.067481, 22.167147, 2.1077573, 1.2813168,
        17.445967, -4.5298853, -3.8749488, -21.857153, -0.33539384, 2.1862486, -1.2261397, -10.763960,
        4.9033306, 0.33745450, 0.89187807, -0.87146227, -21.344961, 6.7876961, 4.8441510, -4.1353252,
        13.148079, -19.310842, 0.23430889, -10.670645, -16.856733, 4.1926396, 10.903613, 6.4191176,
        -6.0338409, -1.6446804, 6.2580204, 7.0709549, 0.48976456, 1.8302092, 1.7804860, -11.791449,
        3.5548148, -4.5864163, -12.271088, -2.1164685, -0.21297715, 3.2345289, -4.1238710, 18.888733,
        2.9010470, 0.52596943, -5.0522649, -2.2022351, 7.1542968, 3.0571825, -4.7109594, -5.5241830,
        -1.2295989, -0.020839160, -3.5810073, 10.262012, 3.3730524, -0.35300787, 5.5544007, -0.49215171,
        -3.0484051, 8.8406339, 4.0829543, -3.6085812, 4.1223855, -2.3146242, 1.4168210, 4.8755271,
        -0.35984407, -2.4964330, -3.8513665, -1.5564510, -1.9459387, -1.3577985, 3.5000945, 1.0252986,
        -1.7508598, 1.7307071, -2.1822014, 1.4139510, -0.26492313, -6.9929332, -1.4905007, -4.5523653,
        -1.8904264, 0.37805394, -3.0843898, 1.4855542, -1.8286614, 1.3366694, 4.1064140, 4.5006266,
        1.7839179, 0.065352770, -1.7194005, -2.3763683, -5.3105012, 2.9516056, 7.6944122, -0.60640483,
        -0.90663647, -18.197919, -0.22268430, -3.4400329, 6.9501785, 7.7956870, 0.0037390682, 10.998891,
        2.9412122, 2.0622571, -15.363395, 4.1556472, 7.0323148, 1.0251932, 0.93672411, 2.1684778,
        0.85332394, -4.2812941, -25.915980, -3.4342963, 3.2791777, 11.406014, -0.65809084, 4.2865970,
        -8.2890036, -9.3558858, -2.9382501, -8.5033111, -0.99220017, -2.8478891, 17.505131, -5.2939784,
        -10.078519, -1.4674728, -2.6179193, 3.5409029, 4.3257668, 3.5148688, 28.178136, 5.2449612,
        -5.0636617, 2.0006635, 1.8238269, -5.6780770, 6.2723766, 4.9277323, 1.7120995, 4.4704590,
        -1.2675444, 1.1493339, -10.753774, 4.3761007, 7.2677427, 0.36360394, 2.8309246, -5.3059011,
        -3.5536583, -1.0870696, -13.639178, -2.3551396, 2.8576233, -5.5331230, -1.5627113, 2.3258584,
        -3.3010578, -0.84873245, 0.0080297553, -2.2476818, 0.46460275, 0.031989059, 5.2853662, -0.61580318,
        -2.8430810, 0.26927965, -1.6725977, 1.4794579, 0.98257246, -0.15173515, 2.8783568, 0.19381363,
        -0.34831400, 2.4890052, 0.51564540, 0.47667340, 0.84665044, -0.0029806413, 0.45702139, -0.16992506,
        0.27108456, -0.22027725, -1.6750958, -0.92050932, 0.36256877, -0.0041920808, 0.52454048, 0.22281138,
        0.039193071, 0.033185478, -0.0030239314, -0.0043899657, -0.0015373169, 0.0090361619, 0.00015367701,
        -0.020585023, -0.0029039688, -0.0029995008
    ])

    bzimf_low, bzimf_high = -10.0, 10.0
    scale_low, scale_high = 1.0, 2.0
    eps_low, eps_high = 0.5, 0.8

    fbz = (bzimf * 2.0 - bzimf_high - bzimf_low) / (bzimf_high - bzimf_low)
    fbz21 = np.abs(fbz)
    fsca = (scale * 2.0 - scale_high - scale_low) / (scale_high - scale_low)
    feps = (eps * 2.0 - eps_high - eps_low) / (eps_high - eps_low)

    a0 = a[nlin] + a[nlin + 2] * fbz + a[nlin + 4] * fbz21 + a[nlin + 6] * fsca + a[nlin + 8] * feps
    b0 = a[nlin + 1] + a[nlin + 3] * fbz + a[nlin + 5] * fbz21 + a[nlin + 7] * fsca + a[nlin + 9] * feps

    fx = 0.0
    fy = 0.0
    fz = 0.0

    ind = 0

    for i in range(1, 5):  # 1 to 4
        for k in range(1, 6):  # 1 to 5
            a_i = a0 * i
            b_k = b0 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)

            bx = xk * ex * cay * sbz
            by = -a_i * ex * say * sbz
            bz = b_k * ex * cay * cbz

            ai = (a[ind] +
                  a[ind + 1] * fbz +
                  a[ind + 2] * fsca +
                  a[ind + 3] * feps +
                  a[ind + 4] * fbz21 +
                  a[ind + 5] * fsca ** 2 +
                  a[ind + 6] * feps ** 2 +
                  a[ind + 7] * fbz * fsca +
                  a[ind + 8] * fsca * feps +
                  a[ind + 9] * fbz * feps +
                  a[ind + 10] * fbz * fbz21 +
                  a[ind + 11] * fsca ** 3 +
                  a[ind + 12] * feps ** 3 +
                  a[ind + 13] * fbz * fsca * feps +
                  a[ind + 14] * fbz21 * fsca +
                  a[ind + 15] * fbz * fsca ** 2 +
                  a[ind + 16] * fsca ** 2 * feps +
                  a[ind + 17] * fsca * feps ** 2 +
                  a[ind + 18] * fbz * feps ** 2 +
                  a[ind + 19] * fbz21 * feps)

            fx += ai * bx
            fy += ai * by
            fz += ai * bz

            ind += 20

    return fx, fy, fz

@jit(fastmath=True, nopython=True)
def prc_unsh_nm(eps, scale, x, y, z):
    dt0 = 0.07  # controls the noon-midnight angular stretch (radians)
    dr0 = 0.17  # controls the magnitude of radial stretch

    # Scaled coordinates
    xsc = x * scale
    ysc = y * scale
    zsc = z * scale

    # Convert Cartesian to spherical
    rhosc = np.sqrt(ysc ** 2 + zsc ** 2)
    rsc = np.sqrt(xsc ** 2 + ysc ** 2 + zsc ** 2)
    t = np.arctan2(rhosc, xsc)
    ts = t - dt0 * rsc ** eps * rhosc / rsc  # deformed theta_s
    dtsdr = -dt0 * eps * rsc ** (eps - 1.0) * rhosc / rsc  # derivative dtheta_s/dr
    dtsdt = 1.0 - dt0 * rsc ** eps * xsc / rsc  # derivative dtheta_s/dtheta

    r_prime = rsc * (1.0 - dr0 * np.sin(t / 2.0) ** 2)
    dr_prime_dr = 1.0 - dr0 * np.sin(t / 2.0) ** 2
    dr_prime_dt = -dr0 * rsc * 0.5 * np.sin(t)

    # Handle small rhosc to avoid division by zero
    if rhosc > 1e-5:
        cp = ysc / rhosc
        sp = zsc / rhosc
        stsst = np.sin(ts) / np.sin(t)
    else:
        cp = 1.0
        sp = 0.0
        stsst = dtsdt

    # Deformed coordinates
    x_s = r_prime * np.cos(ts)
    y_s = r_prime * np.sin(ts) * cp
    z_s = r_prime * np.sin(ts) * sp

    # Call external function to calculate Cartesian B components at primed location
    # Note: prc_nm_undeformed needs to be defined or imported
    bxas, byas, bzas = prc_nm_undeformed(x_s, y_s, z_s)

    # Spherical coordinates at deformed position
    rho2_s = y_s ** 2 + z_s ** 2
    r_s = np.sqrt(rho2_s + x_s ** 2)
    rho_s = np.sqrt(rho2_s)

    ct_s = x_s / r_s
    st_s = rho_s / r_s

    # Convert from Cartesian to spherical magnetic field components
    bras = (x_s * bxas + y_s * byas + z_s * bzas) / r_s
    bthetaas = (byas * cp + bzas * sp) * ct_s - bxas * st_s
    bphias = bzas * cp - byas * sp

    # Apply deformation tensor
    br_s = stsst * dtsdt * (r_prime / rsc) ** 2 * bras - r_prime / rsc ** 2 * stsst * dr_prime_dt * bthetaas
    btheta_s = -r_prime ** 2 / rsc * stsst * dtsdr * bras + stsst * (r_prime / rsc) * dr_prime_dr * bthetaas
    bphi_s = (r_prime / rsc) * (dr_prime_dr * dtsdt - dr_prime_dt * dtsdr) * bphias

    # Convert from spherical to Cartesian
    bxsc = br_s * np.cos(t) - btheta_s * np.sin(t)
    bmerid = br_s * np.sin(t) + btheta_s * np.cos(t)
    bysc = bmerid * cp - bphi_s * sp
    bzsc = bmerid * sp + bphi_s * cp


    return bxsc, bysc, bzsc

@jit(fastmath=True, nopython=True)
def prc_nm_undeformed(x, y, z):
    r, t, p = sphcar(x, y, z, -1)
    br, bt, bp = brbtbp_prc(r, t, p)
    return bspcar(t, p, br, bt, bp)

@jit(fastmath=True, nopython=True)
def brbtbp_prc(r, t, p):
    abr = np.array([
        18.83023014, 468934.6659, -468945.9056, 0.0009310123492, 341.5640336, -341.5724899,
        0.02834275159, 2815.457002, -2815.394238, 4273.405633, 142912593.7, -142912703.7,
        -3.144179212e-09, -9.652596615e-06, 9.661074043e-06, 0.002089306536, -14080.28783,
        14080.43837, 0.1113334263, -0.1062192861, 0.3149232975, 11.36515559, 4.723154529,
        3.984559924, 4.585739484, 4.032706815, 3.741667937, 0.05798779413, 2.417509610,
        8.492134117, 7.761662895, 7.389416256, 7.460193567, 1.110098139, 1.110074621,
        2.821936969, 9.037995133, 5.282888651, 6.262172568, 0.8249294900, 1.195457593,
        2.469064281, 5.071438197, 4.822507104, 0.1568808225, 0.1079224544, 0.1616535726,
        2.078602607, 0.3735552854, 4.279143730, 1.863086547, 1.080625720, 1.104468013,
        6.456298650, 1.970461884, 1.974588436
    ])

    abt = np.array([
        3.356791597, -5.883419687, -2.058543379, 0.0001100303924, -0.001943189958, 0.002793918244,
        0.0002049870425, 0.0005220714000, 0.01019940452, -75.87370859, -47.96110111,
        24.36085698, -5.362635518e-06, 1.925145540e-05, -3.548180534e-06, -0.0003037668516,
        0.0007799787312, -0.01411180236, 0.06526098636, -0.06165246724, 0.7321218781,
        11.63639868, 4.570394366, 5.991114966, 7.610650545, 4.318030252, 4.599681445,
        0.3488975773, 1.700478186, 5.759626972, 1.666470246, 7.067086748, 7.451917780,
        0.4817635507, 2.681223071, 4.434829894, 5.173486046, 0.4858312077, 5.763669434,
        3.765430495, 1.565144414, 2.389620597, 6.040651474, 5.200117858, 0.1443594185,
        0.1078706697, 0.1655199312, 1.108410791, 0.2563543278, 1.647886458, 2.002342423,
        1.537733423, 1.010006109, 2.360742095, 1.907047647, 3.557905953
    ])

    # Persistent variable for one-time initialization
    keep = 0

    if keep == 0:
        keep = 1
        abr[47] = abr[47] * abr[35]  # abr48
        abr[48] = abr[48] * abr[35]  # abr49
        abr[49] = abr[49] * abr[35]  # abr50
        abt[47] = abt[47] * abt[35]  # abt48
        abt[48] = abt[48] * abt[35]  # abt49
        abt[49] = abt[49] * abt[35]  # abt50

    # Trigonometric functions
    st = np.sin(t)
    ct = np.cos(t)
    st2 = st ** 2
    if st2 < 1e-10:  # avoid crashes at Z axis
        st2 = 1e-10
    ct2 = ct ** 2
    sp = np.sin(p)
    cp = np.cos(p)

    # Alpha calculations
    alpha = st2 / r
    if alpha >= 1.0:  # avoid crashes inside Earth
        alpha = 0.9999
    dal_dr = -alpha / r
    dal_dt = 2.0 * st * ct / r

    # Calculate br
    dalpha = abr[18] + abr[19] * (r / 10.0) ** abr[20]
    ddal_dr = abr[19] / 10.0 * abr[20] * (r / 10.0) ** (abr[20] - 1.0)

    br40 = ((r / abr[38]) ** 2 * ct2)
    d1 = (r / abr[21]) ** abr[22] + 1.0 + br40 ** abr[50]

    br41 = ((r / abr[39]) ** 2 * ct2)
    d2 = (r / abr[23]) ** abr[24] + 1.0 + br41 ** abr[51]

    br42 = ((r / abr[40]) ** 2 * ct2)
    d3 = (r / abr[25]) ** abr[26] + 1.0 + br42 ** abr[52]

    br43 = ((r / abr[41]) ** 2 * ct2)
    d4 = (r / abr[27]) ** abr[28] + 1.0 + br43 ** abr[53]

    br44 = ((r / abr[42]) ** 2 * ct2)
    d5 = (r / abr[29]) ** abr[30] + 1.0 + br44 ** abr[54]

    br45 = ((r / abr[43]) ** 2 * ct2)
    d6 = (r / abr[31]) ** abr[32] + 1.0 + br45 ** abr[55]

    sta = (st2 ** 2) ** abr[33]
    stb = (st2 ** 2) ** abr[34]

    # S1 terms
    s1 = abr[0] + abr[1] * sta + abr[2] * stb
    dr1 = 1.0 / d1
    dr2 = dr1 * sta
    dr3 = dr1 * stb

    # F1_46 terms
    alpha01 = abr[44]
    sp46 = np.sqrt((alpha + alpha01) ** 2 + dalpha ** 2)
    sm46 = np.sqrt((alpha - alpha01) ** 2 + dalpha ** 2)
    f1_46 = 2.0 / (sp46 + sm46)
    df1_46_dal = -f1_46 / (sp46 + sm46) * ((alpha - alpha01) / sm46 + (alpha + alpha01) / sp46)
    df1_46_dda = -f1_46 / (sp46 + sm46) * dalpha * (1.0 / sm46 + 1.0 / sp46)

    # S2 terms
    s2 = abr[3] + abr[4] * sta + abr[5] * stb
    dr4 = f1_46 ** abr[47] / d2
    dr5 = dr4 * sta
    dr6 = dr4 * stb

    # F2_47 terms
    alpha02 = abr[45]
    sp47 = np.sqrt((alpha + alpha02) ** 2 + dalpha ** 2)
    sm47 = np.sqrt((alpha - alpha02) ** 2 + dalpha ** 2)
    f2_47 = 2.0 / (sp47 + sm47)
    df2_47_dal = -f2_47 / (sp47 + sm47) * ((alpha - alpha02) / sm47 + (alpha + alpha02) / sp47)
    df2_47_dda = -f2_47 / (sp47 + sm47) * dalpha * (1.0 / sm47 + 1.0 / sp47)

    # S3 terms
    s3 = abr[6] + abr[7] * sta + abr[8] * stb
    dr7 = f2_47 ** abr[36] * alpha ** abr[37] / d3
    dr8 = dr7 * sta
    dr9 = dr7 * stb

    # S4 terms
    s4 = abr[9] + abr[10] * sta + abr[11] * stb
    dr10 = f1_46 ** abr[48] / d4
    dr11 = dr10 * sta
    dr12 = dr10 * stb

    # S5 terms
    s5 = abr[12] + abr[13] * sta + abr[14] * stb
    dr13 = f1_46 ** abr[49] / d5
    dr14 = dr13 * sta
    dr15 = dr13 * stb

    # F3_48 terms
    alpha03 = abr[46]
    sp48 = np.sqrt((alpha + alpha03) ** 2 + dalpha ** 2)
    sm48 = np.sqrt((alpha - alpha03) ** 2 + dalpha ** 2)
    f3_48 = 2.0 / (sp48 + sm48)
    df3_48_dal = -f3_48 / (sp48 + sm48) * ((alpha - alpha03) / sm48 + (alpha + alpha03) / sp48)
    df3_48_dda = -f3_48 / (sp48 + sm48) * dalpha * (1.0 / sm48 + 1.0 / sp48)

    # S6 terms
    s6 = abr[15] + abr[16] * sta + abr[17] * stb
    dr16 = f3_48 ** abr[35] * (1.0 - alpha) ** abr[37] / d6
    dr17 = dr16 * sta
    dr18 = dr16 * stb

    # Calculate betar
    betar = (abr[0] * dr1 + abr[1] * dr2 + abr[2] * dr3 + abr[3] * dr4 + abr[4] * dr5 +
             abr[5] * dr6 + abr[6] * dr7 + abr[7] * dr8 + abr[8] * dr9 + abr[9] * dr10 +
             abr[10] * dr11 + abr[11] * dr12 + abr[12] * dr13 + abr[13] * dr14 + abr[14] * dr15 +
             abr[15] * dr16 + abr[16] * dr17 + abr[17] * dr18)

    br = betar * st * ct * cp

    # Calculate derivative d_betar_dr
    d_betar_dr = (
            -s1 / d1 ** 2 * (abr[22] / abr[21] * (r / abr[21]) ** (abr[22] - 1.0) +
                             abr[50] / abr[38] ** 2 * br40 ** (abr[50] - 1.0) * ct2 * 2.0 * r)
            - s2 / d2 ** 2 * (abr[24] / abr[23] * (r / abr[23]) ** (abr[24] - 1.0) +
                              abr[51] / abr[39] ** 2 * br41 ** (abr[51] - 1.0) * ct2 * 2.0 * r) * f1_46 ** abr[47]
            - s3 / d3 ** 2 * (abr[26] / abr[25] * (r / abr[25]) ** (abr[26] - 1.0) +
                              abr[52] / abr[40] ** 2 * br42 ** (abr[52] - 1.0) * ct2 * 2.0 * r) * f2_47 ** abr[
                36] * alpha ** abr[37]
            - s4 / d4 ** 2 * (abr[28] / abr[27] * (r / abr[27]) ** (abr[28] - 1.0) +
                              abr[53] / abr[41] ** 2 * br43 ** (abr[53] - 1.0) * ct2 * 2.0 * r) * f1_46 ** abr[48]
            - s5 / d5 ** 2 * (abr[30] / abr[29] * (r / abr[29]) ** (abr[30] - 1.0) +
                              abr[54] / abr[42] ** 2 * br44 ** (abr[54] - 1.0) * ct2 * 2.0 * r) * f1_46 ** abr[49]
            - s6 / d6 ** 2 * (abr[32] / abr[31] * (r / abr[31]) ** (abr[32] - 1.0) +
                              abr[55] / abr[43] ** 2 * br45 ** (abr[55] - 1.0) * ct2 * 2.0 * r) * f3_48 ** abr[35] * (
                        1.0 - alpha) ** abr[37]
            + s2 / d2 * abr[47] * f1_46 ** (abr[47] - 1.0) * (df1_46_dal * dal_dr + df1_46_dda * ddal_dr)
            + s3 / d3 * (abr[36] * f2_47 ** (abr[36] - 1.0) * (df2_47_dal * dal_dr + df2_47_dda * ddal_dr) * alpha **
                         abr[37]
                         + f2_47 ** abr[36] * abr[37] * alpha ** (abr[37] - 1.0) * dal_dr)
            + s4 / d4 * abr[48] * f1_46 ** (abr[48] - 1.0) * (df1_46_dal * dal_dr + df1_46_dda * ddal_dr)
            + s5 / d5 * abr[49] * f1_46 ** (abr[49] - 1.0) * (df1_46_dal * dal_dr + df1_46_dda * ddal_dr)
            + s6 / d6 * (abr[35] * f3_48 ** (abr[35] - 1.0) * (df3_48_dal * dal_dr + df3_48_dda * ddal_dr) * (
                1.0 - alpha) ** abr[37]
                         - f3_48 ** abr[35] * abr[37] * (1.0 - alpha) ** (abr[37] - 1.0) * dal_dr)
    )

    # Calculate btheta
    dalpha = abt[18] + abt[19] * (r / 10.0) ** abt[20]
    ddal_dr = abt[19] / 10.0 * abt[20] * (r / 10.0) ** (abt[20] - 1.0)

    d1 = (r / abt[21]) ** abt[22] + 1.0 + ((r / abt[38]) ** 2 * ct2) ** abt[50]
    d2 = (r / abt[23]) ** abt[24] + 1.0 + ((r / abt[39]) ** 2 * ct2) ** abt[51]
    d3 = (r / abt[25]) ** abt[26] + 1.0 + ((r / abt[40]) ** 2 * ct2) ** abt[52]
    d4 = (r / abt[27]) ** abt[28] + 1.0 + ((r / abt[41]) ** 2 * ct2) ** abt[53]
    d5 = (r / abt[29]) ** abt[30] + 1.0 + ((r / abt[42]) ** 2 * ct2) ** abt[54]
    d6 = (r / abt[31]) ** abt[32] + 1.0 + ((r / abt[43]) ** 2 * ct2) ** abt[55]

    sta = (st2 ** 2) ** abt[33]
    stb = (st2 ** 2) ** abt[34]

    # S1 terms
    s1 = abt[0] + abt[1] * sta + abt[2] * stb
    ds1dt = 4.0 * st * ct * (abt[1] * abt[33] * st2 ** (2.0 * abt[33] - 1.0) +
                             abt[2] * abt[34] * st2 ** (2.0 * abt[34] - 1.0))
    dr1 = 1.0 / d1
    dr2 = dr1 * sta
    dr3 = dr1 * stb

    # F1_46 terms
    alpha01 = abt[44]
    sp46 = np.sqrt((alpha + alpha01) ** 2 + dalpha ** 2)
    sm46 = np.sqrt((alpha - alpha01) ** 2 + dalpha ** 2)
    f1_46 = 2.0 / (sp46 + sm46)
    df1_46_dal = -f1_46 / (sp46 + sm46) * ((alpha - alpha01) / sm46 + (alpha + alpha01) / sp46)
    df1_46_dda = -f1_46 / (sp46 + sm46) * dalpha * (1.0 / sm46 + 1.0 / sp46)

    # S2 terms
    s2 = abt[3] + abt[4] * sta + abt[5] * stb
    ds2dt = 4.0 * st * ct * (abt[4] * abt[33] * st2 ** (2.0 * abt[33] - 1.0) +
                             abt[5] * abt[34] * st2 ** (2.0 * abt[34] - 1.0))
    dr4 = f1_46 ** abt[47] / d2
    dr5 = dr4 * sta
    dr6 = dr4 * stb

    # F2_47 terms
    alpha02 = abt[45]
    sp47 = np.sqrt((alpha + alpha02) ** 2 + dalpha ** 2)
    sm47 = np.sqrt((alpha - alpha02) ** 2 + dalpha ** 2)
    f2_47 = 2.0 / (sp47 + sm47)
    df2_47_dal = -f2_47 / (sp47 + sm47) * ((alpha - alpha02) / sm47 + (alpha + alpha02) / sp47)
    df2_47_dda = -f2_47 / (sp47 + sm47) * dalpha * (1.0 / sm47 + 1.0 / sp47)

    # S3 terms
    s3 = abt[6] + abt[7] * sta + abt[8] * stb
    ds3dt = 4.0 * st * ct * (abt[7] * abt[33] * st2 ** (2.0 * abt[33] - 1.0) +
                             abt[8] * abt[34] * st2 ** (2.0 * abt[34] - 1.0))
    dr7 = f2_47 ** abt[36] * alpha ** abt[37] / d3
    dr8 = dr7 * sta
    dr9 = dr7 * stb

    # S4 terms
    s4 = abt[9] + abt[10] * sta + abt[11] * stb
    ds4dt = 4.0 * st * ct * (abt[10] * abt[33] * st2 ** (2.0 * abt[33] - 1.0) +
                             abt[11] * abt[34] * st2 ** (2.0 * abt[34] - 1.0))
    dr10 = f1_46 ** abt[48] / d4
    dr11 = dr10 * sta
    dr12 = dr10 * stb

    # S5 terms
    s5 = abt[12] + abt[13] * sta + abt[14] * stb
    ds5dt = 4.0 * st * ct * (abt[13] * abt[33] * st2 ** (2.0 * abt[33] - 1.0) +
                             abt[14] * abt[34] * st2 ** (2.0 * abt[34] - 1.0))
    dr13 = f1_46 ** abt[49] / d5
    dr14 = dr13 * sta
    dr15 = dr13 * stb

    # F3_48 terms
    alpha03 = abt[46]
    sp48 = np.sqrt((alpha + alpha03) ** 2 + dalpha ** 2)
    sm48 = np.sqrt((alpha - alpha03) ** 2 + dalpha ** 2)
    f3_48 = 2.0 / (sp48 + sm48)
    df3_48_dal = -f3_48 / (sp48 + sm48) * ((alpha - alpha03) / sm48 + (alpha + alpha03) / sp48)
    df3_48_dda = -f3_48 / (sp48 + sm48) * dalpha * (1.0 / sm48 + 1.0 / sp48)

    # S6 terms
    s6 = abt[15] + abt[16] * sta + abt[17] * stb
    ds6dt = 4.0 * st * ct * (abt[16] * abt[33] * st2 ** (2.0 * abt[33] - 1.0) +
                             abt[17] * abt[34] * st2 ** (2.0 * abt[34] - 1.0))
    dr16 = f3_48 ** abt[35] * (1.0 - alpha) ** abt[37] / d6
    dr17 = dr16 * sta
    dr18 = dr16 * stb

    # Calculate bit
    bit = (abt[0] * dr1 + abt[1] * dr2 + abt[2] * dr3 + abt[3] * dr4 + abt[4] * dr5 +
           abt[5] * dr6 + abt[6] * dr7 + abt[7] * dr8 + abt[8] * dr9 + abt[9] * dr10 +
           abt[10] * dr11 + abt[11] * dr12 + abt[12] * dr13 + abt[13] * dr14 + abt[14] * dr15 +
           abt[15] * dr16 + abt[16] * dr17 + abt[17] * dr18)

    bt = bit * cp

    # Calculate derivative d_bit_dt
    d_bit_dt = (
            ds1dt / d1 + s1 / d1 ** 2 * abt[50] * ((r / abt[38]) ** 2 * ct2) ** (abt[50] - 1.0) * (
                r / abt[38]) ** 2 * 2.0 * st * ct
            + (ds2dt / d2 + s2 / d2 ** 2 * abt[51] * ((r / abt[39]) ** 2 * ct2) ** (abt[51] - 1.0) * (
                r / abt[39]) ** 2 * 2.0 * st * ct) * f1_46 ** abt[47]
            + s2 / d2 * abt[47] * f1_46 ** (abt[47] - 1.0) * df1_46_dal * dal_dt
            + (ds3dt / d3 + s3 / d3 ** 2 * abt[52] * ((r / abt[40]) ** 2 * ct2) ** (abt[52] - 1.0) * (
                r / abt[40]) ** 2 * 2.0 * st * ct) * f2_47 ** abt[36] * alpha ** abt[37]
            + s3 / d3 * abt[36] * f2_47 ** (abt[36] - 1.0) * df2_47_dal * dal_dt * alpha ** abt[37]
            + s3 / d3 * abt[37] * f2_47 ** abt[36] * alpha ** (abt[37] - 1.0) * dal_dt
            + (ds4dt / d4 + s4 / d4 ** 2 * abt[53] * ((r / abt[41]) ** 2 * ct2) ** (abt[53] - 1.0) * (
                r / abt[41]) ** 2 * 2.0 * st * ct) * f1_46 ** abt[48]
            + s4 / d4 * abt[48] * f1_46 ** (abt[48] - 1.0) * df1_46_dal * dal_dt
            + (ds5dt / d5 + s5 / d5 ** 2 * abt[54] * ((r / abt[42]) ** 2 * ct2) ** (abt[54] - 1.0) * (
                r / abt[42]) ** 2 * 2.0 * st * ct) * f1_46 ** abt[49]
            + s5 / d5 * abt[49] * f1_46 ** (abt[49] - 1.0) * df1_46_dal * dal_dt
            + (ds6dt / d6 + s6 / d6 ** 2 * abt[55] * ((r / abt[43]) ** 2 * ct2) ** (abt[55] - 1.0) * (
                r / abt[43]) ** 2 * 2.0 * st * ct) * f3_48 ** abt[35] * (1.0 - alpha) ** abt[37]
            + s6 / d6 * abt[35] * f3_48 ** (abt[2] - 1.0) * df3_48_dal * dal_dt * (1.0 - alpha) ** abt[37]
            - s6 / d6 * abt[37] * f3_48 ** abt[35] * (1.0 - alpha) ** (abt[37] - 1.0) * dal_dt
    )

    # Calculate bphi from br and btheta using div B = 0
    bp = -sp * (st * (st * ct * (2.0 * betar + r * d_betar_dr) + d_bit_dt) + bit * ct)

    return br, bt, bp

@jit(fastmath=True, nopython=True)
def sphcar(x1, x2, x3, j):
    if j > 0:
        # Spherical to Cartesian conversion
        r, theta, phi = x1, x2, x3
        sq = r * np.sin(theta)
        x = sq * np.cos(phi)
        y = sq * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    else:
        # Cartesian to Spherical conversion
        x, y, z = x1, x2, x3
        sq = x ** 2 + y ** 2
        r = np.sqrt(sq + z ** 2)
        if sq != 0.0:
            sq = np.sqrt(sq)
            phi = np.arctan2(y, x)
            theta = np.arctan2(sq, z)
            if phi < 0.0:
                phi = phi + 2 * np.pi  # 6.283185307 is 2*pi
        else:
            phi = 0.0
            if z < 0.0:
                theta = np.pi  # 3.141592654
            else:
                theta = 0.0
        return r, theta, phi

@jit(fastmath=True, nopython=True)
def bspcar(theta, phi, br, btheta, bphi):
    # Trigonometric functions
    s = np.sin(theta)
    c = np.cos(theta)
    sf = np.sin(phi)
    cf = np.cos(phi)

    # Calculate Cartesian magnetic field components
    be = br * s + btheta * c
    bx = be * cf - bphi * sf
    by = be * sf + bphi * cf
    bz = br * c - btheta * s
    return bx, by, bz

@jit(fastmath=True, nopython=True)
def prc_shld_nm(x, y, z, bzimf, eps, scale):
    nlin = 240
    ntot = 245

    # Array a with 245 elements
    a = np.array([
        34.181192, -0.29556168, 12.909051, 4.9801298, 6.1500375, -0.74915782, -21.611778, -7.1847034,
        5.6796017, 2.2643727, -0.95787448, -4.9424794, -5.6198272, 1.2624966, 18.578143, -0.030092775,
        1.6880753, -9.2193277, -1.5771469, -5.3423316, -20.261817, 3.2126805, -2.4372304, 1.0275728,
        -3.6717185, 12.767695, 8.1013226, 9.7460894, 7.9191163, 1.8208879, 1.9377176, 0.37277955,
        4.0809641, -1.0439858, -4.4019306, 12.338464, -10.577554, 1.7692719, 2.0392403, 3.7824888,
        -37.258280, 2.9374995, -16.101684, -5.7453565, -10.174126, -11.332124, 6.2470586, 11.112049,
        -0.89338110, 3.8577045, 5.0320009, 2.1298388, -0.56721888, -1.6649850, 1.8637425, 2.4207808,
        -1.2246224, -1.0654469, -7.5198908, -1.7276819, 3.0034762, 0.91545679, 2.7752526, 0.019492507,
        -7.7467291, 0.28431333, 2.9739456, 2.2474879, -1.1848038, 0.31421829, 0.99219624, -1.5681495,
        -1.6979963, 0.13900377, 3.6281564, -8.3170206, 0.73709654, -3.2741704, -0.45948651, -0.94160370,
        -25.037655, -1.0611731, -5.5580580, -4.5030523, -15.515880, -5.1363568, 18.529757, 10.939651,
        -2.7949921, -0.054581881, 11.195992, 3.3436551, 5.0624281, -1.3006263, -5.1364744, 3.5690900,
        -4.2923300, 2.1328852, -5.2467221, -0.13620892, 18.770186, -2.3562018, 5.7961592, -0.63064669,
        -5.5749313, -10.435435, -1.8886772, -5.7032962, -5.7101555, -0.76821310, -1.2209654, -1.5274722,
        -5.3357059, 1.0722311, 8.6030986, -18.215785, 9.5865548, -4.5046592, -1.7123164, -4.1614906,
        17.800578, -2.1978165, 14.218007, 2.3727138, -8.2979498, 9.2359859, 10.684782, -2.2509492,
        -0.60787776, -2.2727511, -1.8043182, 0.73848940, 0.76077508, 1.3688103, 2.5610340, 0.47900536,
        -4.5753533, 2.2541873, 6.4528160, 1.7519095, 9.6375720, -3.9428667, 1.4793671, -0.86386190,
        3.9761522, -10.535133, -4.9330324, -8.1509487, -1.4594917, 0.20336362, -2.8632923, 2.0394247,
        -2.4186769, 0.75471181, 1.5998619, -0.72653280, 5.9104472, 2.1555852, -1.6667320, -2.4244646,
        20.502462, -0.27327685, 8.8452226, 4.7386919, 10.593730, 13.271913, -2.3608862, -13.065089,
        3.5673961, -2.1986219, -28.413638, 0.056823373, -10.554222, 2.2221769, 7.9440106, -4.8503839,
        -0.40088324, 5.6948041, 14.466581, 5.4476484, -14.151370, -1.8888276, -6.6069793, -0.83710456,
        11.437575, 1.4843343, -5.7288563, -3.2866698, 5.1769964, 1.4806221, -0.064790109, 5.4533199,
        5.2463491, -0.70123068, -14.265660, 29.018673, -8.8034709, 10.430212, -1.9187523, 2.8974242,
        -5.5815235, -3.4324352, -12.599610, -2.9010821, 24.864737, -0.74230293, -5.6457389, -9.8136339,
        -2.0992820, -6.7259956, 8.8169633, 3.8670580, 8.8654183, -1.2980093, -29.748927, -4.6201843,
        5.5276324, 12.587564, -10.083196, 5.8797310, 4.8521062, 1.3373527, 0.42719831, 1.1380295,
        3.1629646, 6.0495530, 1.8828022, -1.8822390, -2.0332153, -6.2623047, 0.84218678, -3.9049652,
        -0.59496182, -0.14949346, 4.7583767, -14.009779, 2.8995911, -5.0087497, 2.7360707, 1.3343890,
        0.022048751, 0.061515464, 0.0056954579, -0.0098457765, -0.033742397
    ])

    # Other constants
    bzimf_low, bzimf_high = -10.0, 10.0
    scale_low, scale_high = 0.7, 1.4
    eps_low, eps_high = 0.2, 0.6

    # Normalized parameters
    fbz = (bzimf * 2.0 - bzimf_high - bzimf_low) / (bzimf_high - bzimf_low)
    fbz21 = np.abs(fbz) / np.sqrt(1.0 + a[nlin + 2] * fbz ** 2)
    fsca = (scale * 2.0 - scale_high - scale_low) / (scale_high - scale_low)
    feps = (eps * 2.0 - eps_high - eps_low) / (eps_high - eps_low)

    # Calculate a0 and b0
    a0 = a[nlin] + a[nlin + 3] * fsca
    b0 = a[nlin + 1] + a[nlin + 4] * fsca

    # Initialize output variables
    fx = 0.0
    fy = 0.0
    fz = 0.0

    # Loop variables
    ind = 0  # Python uses 0-based indexing

    # Nested loops
    for i in range(1, 7):  # 1 to 6
        for k in range(1, 5):  # 1 to 4
            a_i = a0 * i
            b_k = b0 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)

            bx = xk * ex * cay * sbz
            by = -a_i * ex * say * sbz
            bz = b_k * ex * cay * cbz

            ai = (a[ind] + a[ind + 1] * fbz + a[ind + 2] * fbz21 + a[ind + 3] * feps +
                  a[ind + 4] * fsca + a[ind + 5] * fsca ** 2 + a[ind + 6] * fsca ** 3 +
                  a[ind + 7] * fbz * fsca + a[ind + 8] * fbz21 * fsca + a[ind + 9] * fbz * fsca ** 2)

            fx += ai * bx
            fy += ai * by
            fz += ai * bz

            ind += 10
    return fx, fy, fz

@jit(fastmath=True, nopython=True)
def prc_unsh_dd(eps, scale, x, y, z):
    dt0 = 0.07  # controls the noon-midnight angular stretch (radians)
    dr0 = 0.17  # controls the magnitude of radial stretch

    # Scaled coordinates
    xsc = x * scale
    ysc = y * scale
    zsc = z * scale

    # Convert Cartesian to spherical
    rhosc = np.sqrt(ysc ** 2 + zsc ** 2)
    rsc = np.sqrt(xsc ** 2 + ysc ** 2 + zsc ** 2)
    t = np.arctan2(rhosc, xsc)
    ts = t - dt0 * rsc ** eps * rhosc / rsc  # deformed theta_s
    dtsdr = -dt0 * eps * rsc ** (eps - 1.0) * rhosc / rsc  # derivative dtheta_s/dr
    dtsdt = 1.0 - dt0 * rsc ** eps * xsc / rsc  # derivative dtheta_s/dtheta

    r_prime = rsc * (1.0 - dr0 * np.sin(t / 2.0) ** 2)
    dr_prime_dr = 1.0 - dr0 * np.sin(t / 2.0) ** 2
    dr_prime_dt = -dr0 * rsc * 0.5 * np.sin(t)

    # Handle small rhosc to avoid division by zero
    if rhosc > 1e-5:
        cp = ysc / rhosc
        sp = zsc / rhosc
        stsst = np.sin(ts) / np.sin(t)
    else:
        cp = 1.0
        sp = 0.0
        stsst = dtsdt

    # Deformed coordinates
    x_s = r_prime * np.cos(ts)
    y_s = r_prime * np.sin(ts) * cp
    z_s = r_prime * np.sin(ts) * sp

    # Call external function to calculate Cartesian B components at primed location
    bxas, byas, bzas = prc_dd_undeformed(x_s, y_s, z_s)

    # Spherical coordinates at deformed position
    rho2_s = y_s ** 2 + z_s ** 2
    r_s = np.sqrt(rho2_s + x_s ** 2)
    rho_s = np.sqrt(rho2_s)

    ct_s = x_s / r_s
    st_s = rho_s / r_s

    # Convert from Cartesian to spherical magnetic field components
    bras = (x_s * bxas + y_s * byas + z_s * bzas) / r_s
    bthetaas = (byas * cp + bzas * sp) * ct_s - bxas * st_s
    bphias = bzas * cp - byas * sp

    # Apply deformation tensor
    br_s = stsst * dtsdt * (r_prime / rsc) ** 2 * bras - r_prime / rsc ** 2 * stsst * dr_prime_dt * bthetaas
    btheta_s = -r_prime ** 2 / rsc * stsst * dtsdr * bras + stsst * (r_prime / rsc) * dr_prime_dr * bthetaas
    bphi_s = (r_prime / rsc) * (dr_prime_dr * dtsdt - dr_prime_dt * dtsdr) * bphias

    # Convert from spherical to Cartesian
    bxsc = br_s * np.cos(t) - btheta_s * np.sin(t)
    bmerid = br_s * np.sin(t) + btheta_s * np.cos(t)
    bysc = bmerid * cp - bphi_s * sp
    bzsc = bmerid * sp + bphi_s * cp

    # Assign final magnetic field components (no scaling)
    bx = bxsc
    by = bysc
    bz = bzsc

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def prc_dd_undeformed(x, y, z):
    x1 = -y
    y1 = x
    r, t, p = sphcar(x1, y1, z, -1)
    br, bt, bp = brbtbp_prc(r, t, p)
    bx1, by1, bz = bspcar(t, p, br, bt, bp)
    bx = by1
    by = -bx1

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def prc_shld_dd(x, y, z, bzimf, eps, scale):
    nlin = 240
    ntot = 245

    # Array a with 245 elements
    a = np.array([
        -173.08748, -12.051841, -19.946824, 19.642193, 70.310018, 75.177656, 34.520150, -24.974531,
        70.732351, 10.096452, -13.597701, 9.3944292, -15.171517, -27.923772, -108.18465, -35.158396,
        89.647468, 37.444494, -61.947256, -1.9842204, 18.894318, -27.206972, 13.294326, 22.073997,
        -1.1570484, -63.182196, 11.914302, -12.687222, 21.617200, -0.37557404, -48.038747, 4.8491468,
        -33.408076, 20.712963, 10.266758, 35.618844, -49.948130, -32.758746, 0.21312420, -4.6602706,
        193.50819, -2.1276278, 59.650791, -10.523259, 18.628808, 18.457484, -32.078623, 1.5128950,
        -26.401573, -16.213012, 9.3973986, 11.421672, -15.995494, 2.5182425, -59.907718, -25.152481,
        27.371634, 21.511491, -44.570481, 2.9882407, 54.043177, -17.106961, 35.876256, -2.5569585,
        46.939193, -42.179131, -1.7869133, -7.4885614, 52.033555, 16.738295, -32.316580, 19.629341,
        -24.408864, -6.3288935, 21.064037, 46.977419, -26.677679, -7.5744088, -0.86427191, 6.0345735,
        5.9322392, 7.8290484, -29.787536, -10.080507, 22.539417, -27.461210, -12.472286, 4.3290604,
        -16.318832, -9.1716953, -75.242932, 8.2176937, -20.877768, 21.847929, 17.448625, 48.630093,
        -41.600240, -19.127657, 25.205628, 3.1041289, 21.523811, 2.0616062, 20.424430, -21.764999,
        29.124129, 13.141096, -19.796480, -2.6500204, 25.988169, 7.7667605, 16.157029, 2.5209401,
        8.8512540, -14.021652, -13.846838, -6.4784611, 37.757105, 28.494851, -20.904987, -5.0475496,
        -105.37290, -0.73661356, -43.981724, 12.185297, -18.062315, -31.225747, -3.4212167, -3.6461840,
        8.1305868, 18.702539, -30.208994, -10.722295, 19.462849, 5.2336031, 18.313843, 53.223035,
        -34.990831, -23.663671, 38.032338, -6.2281490, -20.807831, 13.285530, -26.683761, 3.4193421,
        -49.460525, 15.576843, -16.895695, 6.2609664, -57.219748, -19.128612, 50.075329, -26.916746,
        33.645466, 15.689089, -36.789764, -55.196457, 47.574860, 17.132535, -14.424625, -15.822546,
        83.202588, -5.0383153, 49.946407, -2.4785598, -47.015559, -30.138088, 14.711106, 17.344510,
        -4.0845934, 17.843695, 115.33077, -21.099448, 54.761848, -28.303078, 62.293728, -20.104082,
        58.378709, 7.1908603, 23.665277, -11.362744, -66.391112, 27.684715, -73.050201, 19.355094,
        -17.811321, -5.3336640, 10.516419, 15.630203, -62.763691, -7.6699197, -27.861515, 2.7636923,
        0.82739237, -9.6442883, 40.062154, 29.212481, -47.375480, -36.804083, 64.721840, 32.029313,
        -64.903513, 24.134575, -35.928899, 1.1136802, -34.727863, -11.552687, -28.495564, 18.288438,
        -23.780405, -15.319146, -9.4164136, -8.5964473, -15.217163, 10.599390, 18.596816, 22.026133,
        -1.5248560, -27.218851, -2.4900813, 6.3982362, 24.291437, -10.071961, 40.848718, -9.2288516,
        -4.0876113, -7.8354217, -0.68464361, 6.3643265, 40.134695, 12.120108, 2.9190973, 4.0186331,
        -10.513156, 2.9362587, -7.4585215, -2.5039220, 8.5429537, 9.3137662, -29.772613, -15.041690,
        0.028940081, 0.048745244, 0.27521861, 0.000019257827, -0.030864086
    ])

    # Other constants
    bzimf_low, bzimf_high = -10.0, 10.0
    scale_low, scale_high = 0.7, 1.4
    eps_low, eps_high = 0.2, 0.6

    # Normalized parameters
    fbz = (bzimf * 2.0 - bzimf_high - bzimf_low) / (bzimf_high - bzimf_low)
    fbz21 = np.abs(fbz) / np.sqrt(1.0 + a[nlin + 2] * fbz ** 2)
    fsca = (scale * 2.0 - scale_high - scale_low) / (scale_high - scale_low)
    feps = (eps * 2.0 - eps_high - eps_low) / (eps_high - eps_low)

    # Calculate a0 and b0
    a0 = a[nlin] + a[nlin + 3] * fsca
    b0 = a[nlin + 1] + a[nlin + 4] * fsca

    # Initialize output variables
    fx = 0.0
    fy = 0.0
    fz = 0.0

    # Loop variables
    ind = 0  # Python uses 0-based indexing

    # Nested loops
    for i in range(1, 7):  # 1 to 6
        for k in range(1, 5):  # 1 to 4
            a_i = a0 * i
            b_k = b0 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)

            bx = xk * ex * say * sbz
            by = a_i * ex * cay * sbz
            bz = b_k * ex * say * cbz

            ai = (a[ind] + a[ind + 1] * fbz + a[ind + 2] * fbz21 + a[ind + 3] * feps +
                  a[ind + 4] * fsca + a[ind + 5] * fsca ** 2 + a[ind + 6] * fsca ** 3 +
                  a[ind + 7] * fbz * fsca + a[ind + 8] * fbz21 * fsca + a[ind + 9] * fbz * fsca ** 2)

            fx += ai * bx
            fy += ai * by
            fz += ai * bz

            ind += 10
    return fx, fy, fz

@jit(fastmath=True, nopython=True)
def prcs_unsh(eps, scale, x, y, z):
    dt0 = 0.07  # controls the noon-midnight angular stretch (radians)
    dr0 = 0.17  # controls the magnitude of radial stretch

    # Scaled coordinates
    xsc = x * scale
    ysc = y * scale
    zsc = z * scale

    # Convert Cartesian to spherical
    rhosc = np.sqrt(ysc ** 2 + zsc ** 2)
    rsc = np.sqrt(xsc ** 2 + ysc ** 2 + zsc ** 2)
    t = np.arctan2(rhosc, xsc)
    ts = t - dt0 * rsc ** eps * rhosc / rsc  # deformed theta_s
    dtsdr = -dt0 * eps * rsc ** (eps - 1.0) * rhosc / rsc  # derivative dtheta_s/dr
    dtsdt = 1.0 - dt0 * rsc ** eps * xsc / rsc  # derivative dtheta_s/dtheta

    r_prime = rsc * (1.0 - dr0 * np.sin(t / 2.0) ** 2)
    dr_prime_dr = 1.0 - dr0 * np.sin(t / 2.0) ** 2
    dr_prime_dt = -dr0 * rsc * 0.5 * np.sin(t)

    # Handle small rhosc to avoid division by zero
    if rhosc > 1e-5:
        cp = ysc / rhosc
        sp = zsc / rhosc
        stsst = np.sin(ts) / np.sin(t)
    else:
        cp = 1.0
        sp = 0.0
        stsst = dtsdt

    # Deformed coordinates
    x_s = r_prime * np.cos(ts)
    y_s = r_prime * np.sin(ts) * cp
    z_s = r_prime * np.sin(ts) * sp

    # Calculate Cartesian B components at primed location
    bxas, byas, bzas = prcs_axisymmetric(x_s, y_s, z_s)

    # Spherical coordinates at deformed position
    rho2_s = y_s ** 2 + z_s ** 2
    r_s = np.sqrt(rho2_s + x_s ** 2)
    rho_s = np.sqrt(rho2_s)

    ct_s = x_s / r_s
    st_s = rho_s / r_s

    # Convert from Cartesian to spherical magnetic field components
    bras = (x_s * bxas + y_s * byas + z_s * bzas) / r_s
    bthetaas = (byas * cp + bzas * sp) * ct_s - bxas * st_s
    bphias = bzas * cp - byas * sp

    # Apply deformation tensor
    br_s = stsst * dtsdt * (r_prime / rsc) ** 2 * bras - r_prime / rsc ** 2 * stsst * dr_prime_dt * bthetaas
    btheta_s = -r_prime ** 2 / rsc * stsst * dtsdr * bras + stsst * (r_prime / rsc) * dr_prime_dr * bthetaas
    bphi_s = (r_prime / rsc) * (dr_prime_dr * dtsdt - dr_prime_dt * dtsdr) * bphias

    # Convert from spherical to Cartesian
    bxsc = br_s * np.cos(t) - btheta_s * np.sin(t)
    bmerid = br_s * np.sin(t) + btheta_s * np.cos(t)
    bysc = bmerid * cp - bphi_s * sp
    bzsc = bmerid * sp + bphi_s * cp
    return bxsc, bysc, bzsc

@jit(fastmath=True, nopython=True)
def prcs_axisymmetric(x, y, z):
    f1, f2 = -77.8729, 63.2601  # magnitudes of outer and inner loops
    dr = 2.60373  # damping scale radius
    r1, d1 = 5.49124, 2.80109  # outer loop radius and spread
    r2, d2 = 4.27404, 3.79450  # inner loop radius and spread
    al = 0.537641  # factor in Fourier terms
    a1, a2, a3, a4, a5, a6, a7 = 0.611891, -0.765870, 4.75619, -5.74791, 0.344349, -1.10502, 1.32319
    b1, b2, b3, b4, b5, b6, b7 = 6.24487, -2.47322, 3.80501, -12.3363, 0.882129, -0.890554, 3.31365

    # Calculate rho (radial distance in xy-plane)
    rho = np.sqrt(x ** 2 + y ** 2)
    if rho < 1e-9:  # prevent division by zero at z-axis
        rho = 1e-9

    # Deformation terms for rho
    rr2 = ((rho - r1) ** 2 + z ** 2) / dr ** 2
    dex = np.exp(-rr2)
    argr = al * (rho - r1) / dr
    argz = al * z / dr

    sr = np.sin(argr)
    cr = np.cos(argr)
    s2r = 2.0 * sr * cr
    c2r = cr ** 2 - sr ** 2
    sz = np.sin(argz)
    cz = np.cos(argz)
    s2z = 2.0 * sz * cz
    c2z = cz ** 2 - sz ** 2

    sumrho = a1 + a2 * sr + a3 * cr + a4 * cz + a5 * s2r + a6 * c2r + a7 * c2z
    brackrho = 1.0 + dex * sumrho
    rho_ast = rho * brackrho

    # Deformation terms for z
    sumz = b1 + b2 * sr + b3 * cr + b4 * cz + b5 * s2r + b6 * c2r + b7 * c2z
    brackz = 1.0 + dex * sumz
    z_ast = z * brackz

    # Derivatives for deformation tensor
    drhoast_drho = brackrho - rho * dex * sumrho * 2.0 / dr ** 2 * (rho - r1) + \
                   rho * dex * (a2 * cr - a3 * sr + 2.0 * a5 * c2r - 2.0 * a6 * s2r) * al / dr
    drhoast_dz = -rho * dex * 2.0 * z / dr ** 2 * sumrho - rho * dex * (a4 * np.sin(argz) + 2.0 * a7 * s2z) * al / dr
    dzast_drho = -z * 2.0 * (rho - r1) / dr ** 2 * dex * sumz + z * dex * (
                b2 * cr - b3 * sr + 2.0 * b5 * c2r - 2.0 * b6 * s2r) * al / dr
    dzast_dz = brackz - z * dex * 2.0 * z / dr ** 2 * sumz - z * dex * (b4 * np.sin(argz) + 2.0 * b7 * s2z) * al / dr

    # Deformed Cartesian coordinates
    x_ast = rho_ast * x / rho
    y_ast = rho_ast * y / rho

    # First (outer) loop magnetic field
    bx_ast, by_ast, bz_ast = spread_loop_b(r1, d1, x_ast, y_ast, z_ast)

    # Radial component in deformed coordinates
    # Note: rho_ast < 1e-9 check is commented out in Fortran; assuming it's handled in spread_loop_b
    brho_ast = (bx_ast * x_ast + by_ast * y_ast) / rho_ast

    # Apply deformation tensor (assuming axisymmetric, Bphi_s = 0)
    brho_s = rho_ast / rho * (dzast_dz * brho_ast - drhoast_dz * bz_ast)
    bphi_s = 0.0

    # Convert to Cartesian for outer loop
    bx1 = brho_s * x / rho
    by1 = brho_s * y / rho
    bz1 = rho_ast / rho * (-dzast_drho * brho_ast + drhoast_drho * bz_ast)

    # Second (inner) loop magnetic field (no deformation applied)
    bx2, by2, bz2 = spread_loop_b(r2, d2, x, y, z)

    # Combine contributions
    bx = f1 * bx1 + f2 * bx2
    by = f1 * by1 + f2 * by2
    bz = f1 * bz1 + f2 * bz2

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def prcs_shld(x, y, z, bzimf, eps, scale):
    nlin = 400
    ntot = 410

    # Array a with 410 elements
    a = np.array([
        67.092921, 13.270625, 34.938796, -53.019097, -54.744225, -142.90710, 35.185473, 99.929273,
        -127.46201, -7.0264969, -3.2734459, -275.99991, -3.8623252, -12.664719, -72.253859, 116.35440,
        -52.411422, 54.432181, 17.371851, 7.3617241, -164.38164, 165.84226, -711.83825, 113.36296,
        109.42072, -929.51327, -110.82952, 434.17477, 288.03817, 6.1706649, 9.7718739, -286.24759,
        -14.015289, -2.1008396, 126.05752, 282.59248, 142.69314, -123.66749, -31.802267, -15.640632,
        238.58112, -254.66810, 1257.9457, -112.44743, -59.230558, 1924.9509, 148.90583, -694.01152,
        -493.48477, 56.923070, 1.4569275, 667.42887, -41.747721, 104.37560, 3.7149568, -485.23713,
        -343.89426, 191.90338, 24.901160, 24.788761, 351.70223, 140.88694, 2425.7980, -56.891302,
        16.668083, 4094.5574, 29.902185, 273.67164, -58.838142, -178.49893, -22.536087, 2163.2304,
        44.363115, -248.24211, -51.199528, 160.71389, -36.753723, 20.010564, 10.209530, -30.461139,
        -555.52696, 15.720341, -3010.1245, 155.78289, 9.4652458, -4483.3727, -70.346002, 147.03345,
        432.46840, 121.64489, 20.893073, -2048.0019, -18.382855, 152.15268, 40.726342, 130.76979,
        296.10647, -73.339947, -15.670729, 12.525128, -148.40733, -45.025323, -187.75145, 212.25210,
        127.38564, 333.89531, -57.797716, -293.64770, 297.52639, 26.268864, 13.531224, 474.23566,
        -40.318562, 29.233667, 198.49372, -312.27491, 28.741175, -52.658778, -40.219051, -39.185930,
        412.91117, -394.01179, 1737.5872, -382.88757, -269.96948, 2181.4441, 244.20319, -1056.5669,
        -726.82929, -49.985499, -35.686778, 765.27272, 67.791703, -28.973495, -357.33382, -688.56771,
        -278.32442, 251.40576, 69.814287, 59.423655, -634.02416, 619.71045, -3408.0255, 406.68820,
        180.48161, -5000.3553, -384.92336, 1812.8664, 1341.3575, -54.334433, 29.524610, -1819.0054,
        51.508838, -145.15971, 111.73354, 1286.9907, 866.98973, -459.83948, -51.909339, -64.481897,
        -555.23446, -319.93675, -3875.3579, -6.3196170, -66.827324, -6650.5678, 16.409902, -737.03112,
        -188.99819, 311.36561, 21.378528, -3596.4971, -78.343923, 444.51971, 40.793612, -468.43967,
        -121.06795, 37.939083, -20.634841, 63.934436, 1062.0842, -36.231464, 5640.0796, -280.63113,
        -13.761969, 8320.7541, 114.27974, -287.43254, -772.14426, -229.88876, -35.823631, 3780.9988,
        35.001138, -288.58457, -70.257472, -251.88512, -527.38381, 118.33572, 31.073828, -24.603263,
        157.95496, 51.574378, 69.465668, -271.30470, -106.37902, -432.71333, -40.600744, 299.96534,
        -181.99764, -15.791806, -17.181854, -316.28675, 62.613729, -7.4909394, -182.02324, 314.49164,
        143.26692, -60.560446, 35.185922, 69.169120, -364.95751, 335.52941, -1258.0890, 444.73661,
        225.99453, -1381.2258, -112.63644, 921.79308, 564.95350, 53.532936, 40.431394, -510.20127,
        -93.203882, 31.149961, 335.70634, 592.42622, 63.162396, -102.91484, -55.945556, -88.240443,
        589.43660, -550.21307, 2931.5956, -479.17049, -182.91345, 3969.1557, 289.24339, -1703.2794,
        -1202.2664, -15.543649, -50.112319, 1426.0228, 6.8537302, 41.780365, -197.25998, -1216.6085,
        -677.78276, 326.53483, 38.915142, 68.212804, 190.15509, 270.59525, 1576.6995, 143.22388,
        83.065752, 2980.7788, -72.020515, 746.70456, 439.51349, -156.07429, 10.068961, 1747.1771,
        35.646634, -236.18671, 44.925005, 508.14470, 261.56079, -85.193561, 12.283103, -47.461691,
        -625.55547, 21.651348, -3248.9452, 144.15061, 1.4834306, -4752.9412, -50.012899, 155.94362,
        396.74042, 134.22233, 16.761398, -2153.6872, -19.913973, 169.75862, 30.967640, 135.65216,
        273.18686, -51.520718, -19.612647, 15.786038, -62.447098, -17.317582, 87.732421, 115.16453,
        38.086845, 237.52137, 83.035632, -107.97703, 5.8965002, -4.8979104, 6.6369115, 58.059233,
        -22.700805, -15.486878, 62.355929, -123.77219, -129.11985, 90.434149, -10.945588, -39.232726,
        115.41409, -113.53586, 194.56040, -173.39925, -69.224744, 160.44437, -51.477195, -301.11182,
        -105.44010, -6.7595220, -12.648877, 113.10476, 35.664527, 8.0166560, -107.38371, -176.73792,
        89.043697, -56.968075, 17.048559, 46.443231, -185.87547, 188.86931, -757.84943, 179.22860,
        60.823743, -937.96674, -35.645236, 585.02278, 331.76126, 10.288079, 17.843489, -340.74042,
        -14.476700, -6.5628946, 79.762834, 408.86717, 139.10040, -40.785905, -11.651632, -29.496072,
        8.2459758, -92.382966, -134.30925, -76.270449, -30.817667, -390.23429, 19.971322, -280.24610,
        -180.38338, 24.392052, -8.2361709, -275.73457, -2.7358093, 41.981227, -31.387494, -195.23967,
        -96.673039, 22.050945, -1.9334181, 14.189961, 120.58276, -1.2455097, 619.68902, -20.326637,
        2.2146868, 906.01420, 6.5408356, -17.214191, -59.380466, -26.169239, -1.9754983, 411.94515,
        3.4939633, -33.610588, -2.2437028, -16.302879, -43.122770, 6.7976165, 4.2049134, -3.7087596,
        0.021978915, 0.049592622, -0.000036118249, -0.0058714729, 0.0042928510, 0.0060047340,
        -0.029960545, -0.038787015, 0.00047028426, -0.0036136501
    ])

    # Other constants
    bzimf_low, bzimf_high = -10.0, 10.0
    scale_low, scale_high = 0.5, 1.5
    eps_low, eps_high = -1.0, 1.0

    # Normalized parameters
    fbz = (bzimf * 2.0 - bzimf_high - bzimf_low) / (bzimf_high - bzimf_low)
    fbz21 = np.abs(fbz)
    fsca = (scale * 2.0 - scale_high - scale_low) / (scale_high - scale_low)
    feps = (eps * 2.0 - eps_high - eps_low) / (eps_high - eps_low)

    # Calculate a0 and b0
    a0 = a[nlin] + a[nlin + 2] * fbz + a[nlin + 4] * fbz21 + a[nlin + 6] * fsca + a[nlin + 8] * feps
    b0 = a[nlin + 1] + a[nlin + 3] * fbz + a[nlin + 5] * fbz21 + a[nlin + 7] * fsca + a[nlin + 9] * feps

    # Initialize output variables
    fx = 0.0
    fy = 0.0
    fz = 0.0

    # Loop variables
    ind = 0  # Python uses 0-based indexing

    # Nested loops
    for i in range(1, 5):  # 1 to 4
        for k in range(1, 6):  # 1 to 5
            a_i = a0 * i
            b_k = b0 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)

            bx = xk * ex * cay * sbz
            by = -a_i * ex * say * sbz
            bz = b_k * ex * cay * cbz

            ai = (a[ind] + a[ind + 1] * fbz + a[ind + 2] * fsca + a[ind + 3] * feps + a[ind + 4] * fbz21 +
                  a[ind + 5] * fsca ** 2 + a[ind + 6] * feps ** 2 + a[ind + 7] * fbz * fsca + a[ind + 8] * fsca * feps +
                  a[ind + 9] * fbz * feps + a[ind + 10] * fbz * fbz21 + a[ind + 11] * fsca ** 3 + a[
                      ind + 12] * feps ** 3 +
                  a[ind + 13] * fbz * fsca * feps + a[ind + 14] * fbz21 * fsca + a[ind + 15] * fbz * fsca ** 2 +
                  a[ind + 16] * fsca ** 2 * feps + a[ind + 17] * fsca * feps ** 2 + a[ind + 18] * fbz * feps ** 2 +
                  a[ind + 19] * fbz21 * feps)

            fx += ai * bx
            fy += ai * by
            fz += ai * bz

            ind += 20
    return fx, fy, fz

@jit(fastmath=True, nopython=True)
def r1_fac_r(theta00, dtheta00, ps, x, y, z):
    rr = np.array([0.0, 1.0, 1.9529, 3.0653, 4.4573, 6.0765, 7.9188, 10.0630, 12.8026, 17.2532,
                   32.2532, 47.2532, 62.2532, 77.2532, 92.2532])

    # Common block variables (assumed to be global or passed elsewhere)
    curdphi = 0.0
    dd = np.zeros(15)
    sp = np.zeros(25)
    cp = np.zeros(25)
    xxn = np.zeros((15, 25))
    yyn = np.zeros((15, 25))
    zzn = np.zeros((15, 25))
    xxs = np.zeros((15, 25))
    yys = np.zeros((15, 25))
    zzs = np.zeros((15, 25))

    # Constants
    rh0 = 8.0
    alpha = 4.0
    pi = 3.14159265359
    d0 = 0.08
    n = 3
    mw = 25
    t0, dt0, ps0 = 10.0, 10.0, 10.0

    # Initialization check
    if theta00 != t0 or dtheta00 != dt0 or ps != ps0:
        t0 = theta00
        dt0 = dtheta00
        ps0 = ps

        dphi = 6.283185307 / mw
        curdphi = dphi / 2.0

        cc = 1000.0 / 63.712

        for i in range(15):
            dd[i] = d0 * rr[i] * np.sqrt(rr[i])

        sps = np.sin(ps)
        cps = np.cos(ps)

        for k in range(mw):
            phi = (k + 0.5) * dphi
            cp[k] = np.cos(phi)
            sp[k] = np.sin(phi)

            theta0n = theta00 + dtheta00 * np.sin(phi * 0.5) ** 2 - 0.07 * ps * cp[k]
            theta0s = (pi - theta00) - dtheta00 * np.sin(phi * 0.5) ** 2 - 0.07 * ps * cp[k]

            for i in range(15):
                r = rr[i]
                f = r ** n + 1.0 / (np.sin(theta0n)) ** (2 * n) - 1.0
                fn = f ** (1.0 / n)
                stheta = np.sqrt(r / fn)
                rhosm = r * stheta
                if i == 0:
                    theta_s = np.arcsin(stheta)
                else:
                    h = rh0 * sps / cps * ((1.0 + (rhosm / rh0) ** alpha) ** (1.0 / alpha) - 1.0)
                    theta_s = np.arcsin(stheta) + h / r * stheta
                rhosm_s = r * np.sin(theta_s)
                xxn[i, k] = rhosm_s * cp[k]
                yyn[i, k] = rhosm_s * sp[k]
                zzn[i, k] = r * np.cos(theta_s)

                f = r ** n + 1.0 / (np.sin(theta0s)) ** (2 * n) - 1.0
                fn = f ** (1.0 / n)
                stheta = np.sqrt(r / fn)
                rhosm = r * stheta
                if i == 0:
                    theta_s = pi - np.arcsin(stheta)
                else:
                    h = rh0 * sps / cps * ((1.0 + (rhosm / rh0) ** alpha) ** (1.0 / alpha) - 1.0)
                    theta_s = pi - np.arcsin(stheta) + h / r * stheta
                rhosm_s = r * np.sin(theta_s)
                xxs[i, k] = rhosm_s * cp[k]
                yys[i, k] = rhosm_s * sp[k]
                zzs[i, k] = r * np.cos(theta_s)

    # Main calculation
    bxsm = 0.0
    bysm = 0.0
    bzsm = 0.0

    xsm = x * cps - z * sps
    ysm = y
    zsm = z * cps + x * sps

    for k in range(mw):
        # Northern wire
        l = 0
        x1 = xxn[l, k]
        y1 = yyn[l, k]
        z1 = zzn[l, k]
        d1 = dd[l]

        hx = 0.0
        hy = 0.0
        hz = 0.0

        for i in range(1, 15):
            l += 1
            x2 = xxn[l, k]
            y2 = yyn[l, k]
            z2 = zzn[l, k]
            d2 = dd[l]

            x1x2 = x1 - x2
            y1y2 = y1 - y2
            z1z2 = z1 - z2
            xx1 = xsm - x1
            yy1 = ysm - y1
            zz1 = zsm - z1
            d2d1 = d2 - d1

            a = x1x2 ** 2 + y1y2 ** 2 + z1z2 ** 2 + d2d1 ** 2
            b = 2.0 * (xx1 * x1x2 + yy1 * y1y2 + zz1 * z1z2 + d1 * d2d1)
            c = xx1 ** 2 + yy1 ** 2 + zz1 ** 2 + d1 ** 2

            factor = 2.0 * cc * (np.sqrt(1.0 / (a + b + c)) / (2.0 * a + b + 2.0 * np.sqrt(a * (a + b + c))) -
                                 np.sqrt(1.0 / c) / (b + 2.0 * np.sqrt(a * c)))

            hx += factor * (y1 * (zsm - z2) - ysm * z1z2 - y2 * zz1)
            hy += factor * (z1 * (xsm - x2) - zsm * x1x2 - z2 * xx1)
            hz += factor * (x1 * (ysm - y2) - xsm * y1y2 - x2 * yy1)

            x1 = x2
            y1 = y2
            z1 = z2
            d1 = d2

        s = sp[k]  # regular R1 mode
        bxsm += hx * s
        bysm += hy * s
        bzsm += hz * s

        # Southern wire
        l = 0
        x1 = xxs[l, k]
        y1 = yys[l, k]
        z1 = zzs[l, k]
        d1 = dd[l]

        hx = 0.0
        hy = 0.0
        hz = 0.0

        for i in range(1, 15):
            l += 1
            x2 = xxs[l, k]
            y2 = yys[l, k]
            z2 = zzs[l, k]
            d2 = dd[l]

            x1x2 = x1 - x2
            y1y2 = y1 - y2
            z1z2 = z1 - z2
            xx1 = xsm - x1
            yy1 = ysm - y1
            zz1 = zsm - z1
            d2d1 = d2 - d1

            a = x1x2 ** 2 + y1y2 ** 2 + z1z2 ** 2 + d2d1 ** 2
            b = 2.0 * (xx1 * x1x2 + yy1 * y1y2 + zz1 * z1z2 + d1 * d2d1)
            c = xx1 ** 2 + yy1 ** 2 + zz1 ** 2 + d1 ** 2

            factor = 2.0 * cc * (np.sqrt(1.0 / (a + b + c)) / (2.0 * a + b + 2.0 * np.sqrt(a * (a + b + c))) -
                                 np.sqrt(1.0 / c) / (b + 2.0 * np.sqrt(a * c)))

            hx += factor * (y1 * (zsm - z2) - ysm * z1z2 - y2 * zz1)
            hy += factor * (z1 * (xsm - x2) - zsm * x1x2 - z2 * xx1)
            hz += factor * (x1 * (ysm - y2) - xsm * y1y2 - x2 * yy1)

            x1 = x2
            y1 = y2
            z1 = z2
            d1 = d2

        s = sp[k]  # regular R1 mode
        bxsm += hx * s
        bysm += hy * s
        bzsm += hz * s

    bx = (bxsm * cps + bzsm * sps) * curdphi
    by = bysm * curdphi
    bz = (bzsm * cps - bxsm * sps) * curdphi

    return bx, by, bz, curdphi, dd, sp, cp, xxn, yyn, zzn, xxs, yys, zzs

@jit(fastmath=True, nopython=True)
def r1_fac_a(ps, x, y, z, curdphi, dd, sp, cp, xxn, yyn, zzn, xxs, yys, zzs):
    mw = 25
    cc = 15.69563

    sps = np.sin(ps)
    cps = np.cos(ps)

    bxsm = 0.0
    bysm = 0.0
    bzsm = 0.0

    xsm = x * cps - z * sps
    ysm = y
    zsm = z * cps + x * sps

    for k in range(mw):
        # Northern wire
        l = 0
        x1 = xxn[l, k]
        y1 = yyn[l, k]
        z1 = zzn[l, k]
        d1 = dd[l]

        hx = 0.0
        hy = 0.0
        hz = 0.0

        for i in range(1, 15):
            l += 1
            x2 = xxn[l, k]
            y2 = yyn[l, k]
            z2 = zzn[l, k]
            d2 = dd[l]

            x1x2 = x1 - x2
            y1y2 = y1 - y2
            z1z2 = z1 - z2
            xx1 = xsm - x1
            yy1 = ysm - y1
            zz1 = zsm - z1
            d2d1 = d2 - d1

            a = x1x2 ** 2 + y1y2 ** 2 + z1z2 ** 2 + d2d1 ** 2
            b = 2.0 * (xx1 * x1x2 + yy1 * y1y2 + zz1 * z1z2 + d1 * d2d1)
            c = xx1 ** 2 + yy1 ** 2 + zz1 ** 2 + d1 ** 2

            factor = 2.0 * cc * (np.sqrt(1.0 / (a + b + c)) / (2.0 * a + b + 2.0 * np.sqrt(a * (a + b + c))) -
                                 np.sqrt(1.0 / c) / (b + 2.0 * np.sqrt(a * c)))

            hx += factor * (y1 * (zsm - z2) - ysm * z1z2 - y2 * zz1)
            hy += factor * (z1 * (xsm - x2) - zsm * x1x2 - z2 * xx1)
            hz += factor * (x1 * (ysm - y2) - xsm * y1y2 - x2 * yy1)

            x1 = x2
            y1 = y2
            z1 = z2
            d1 = d2

        s = sp[k]  # regular R1 mode
        bxsm += hx * s
        bysm += hy * s
        bzsm += hz * s

        # Southern wire
        l = 0
        x1 = xxs[l, k]
        y1 = yys[l, k]
        z1 = zzs[l, k]
        d1 = dd[l]

        hx = 0.0
        hy = 0.0
        hz = 0.0

        for i in range(1, 15):
            l += 1
            x2 = xxs[l, k]
            y2 = yys[l, k]
            z2 = zzs[l, k]
            d2 = dd[l]

            x1x2 = x1 - x2
            y1y2 = y1 - y2
            z1z2 = z1 - z2
            xx1 = xsm - x1
            yy1 = ysm - y1
            zz1 = zsm - z1
            d2d1 = d2 - d1

            a = x1x2 ** 2 + y1y2 ** 2 + z1z2 ** 2 + d2d1 ** 2
            b = 2.0 * (xx1 * x1x2 + yy1 * y1y2 + zz1 * z1z2 + d1 * d2d1)
            c = xx1 ** 2 + yy1 ** 2 + zz1 ** 2 + d1 ** 2

            factor = -2.0 * cc * (np.sqrt(1.0 / (a + b + c)) / (2.0 * a + b + 2.0 * np.sqrt(a * (a + b + c))) -
                                  np.sqrt(1.0 / c) / (b + 2.0 * np.sqrt(a * c)))

            hx += factor * (y1 * (zsm - z2) - ysm * z1z2 - y2 * zz1)
            hy += factor * (z1 * (xsm - x2) - zsm * x1x2 - z2 * xx1)
            hz += factor * (x1 * (ysm - y2) - xsm * y1y2 - x2 * yy1)

            x1 = x2
            y1 = y2
            z1 = z2
            d1 = d2

        s = sp[k]  # regular R1 mode
        bxsm += hx * s
        bysm += hy * s
        bzsm += hz * s

    bx = (bxsm * cps + bzsm * sps) * curdphi
    by = bysm * curdphi
    bz = (bzsm * cps - bxsm * sps) * curdphi

    return bx, by, bz

@jit(fastmath=True, nopython=True)
def r1_r_shld(x, y, z, psi, bzimf, th, dth):
    nlin = 600
    ntot = 620

    a = np.array([
        -195.37843, 4.0415972, 4.7113514, -3.8744123, -40.288346, -44.933970, -34.771299, -4.1392825,
        -9.1949407, 3.4375187, -14.690003, -1.8278654, 9.4635223, 2.1688159, 4.4106812, -14.075658,
        -1.2132597, -3.5920627, -6.8408246, -0.050219606, 35.895236, -10.532599, -11.343871, -1.4318168,
        39.820113, 41.722271, 34.113998, 6.8919814, 10.032531, -3.2120097, -36.498065, 6.9859034,
        -15.980445, -7.4590678, -38.228081, -27.749225, -17.878699, -2.3552159, 1.9604603, -1.8444788,
        -37.316416, 0.25552844, 14.258028, 4.6653077, -7.4586208, -5.5528448, -10.755532, -2.7538461,
        1.1774179, 0.11173789, 68.567426, -12.392400, -7.0841859, 0.30683078, 21.324141, 3.9179089,
        14.375375, -1.8236683, -6.6669880, -0.22665402, -76.185215, 13.953954, -19.777938, -11.825662,
        -41.817962, -7.2223298, -26.783314, 0.18609890, 12.613857, 1.2118887, 10.144822, 2.5573858,
        15.731561, 2.7489696, 1.9420971, 3.8564871, -0.090946907, -2.8324736, 0.12687770, 0.57948952,
        29.978512, -8.2864692, -13.840230, -2.7516834, 5.8150059, -6.3457287, 4.8497116, 0.87199975,
        -5.7967093, 0.21239467, -47.806040, 1.8699726, 2.6479197, 0.054729405, -8.9777607, 2.9553405,
        -5.9843176, 4.6168902, 2.7046737, 0.53103741, 18.580036, -6.0476852, 3.1427676, 6.8490488,
        27.053706, 20.283200, 15.715324, 0.56613706, 2.1623511, 0.33961763, 32.100088, -4.8866117,
        12.924253, 5.8920024, 26.630711, 18.238599, 14.861890, -1.2331960, 0.86930894, 0.71662766,
        54.826855, 0.18389778, -4.9380264, 0.23768713, 22.957604, 26.044418, 20.478867, 3.8723309,
        5.8427036, -2.1745528, -19.453570, 8.1466283, -7.4827892, -3.5858094, -17.503040, -5.2623141,
        -7.1810672, 0.45213642, 4.6693435, -0.93436551, -3.8183290, 0.96060501, 18.627278, 7.3076167,
        0.69197192, -11.809902, -5.1586410, -3.2005277, -7.0936441, 0.94416264, 47.301815, -8.0711451,
        -4.5154071, 1.6425972, 22.686540, 8.5703158, 14.055783, 1.0379988, -5.1312304, 0.10898832,
        13.112895, 4.1428441, -11.163097, -2.8483741, 7.6506500, 17.113210, 8.0514543, 2.2710154,
        7.9103326, 0.39080202, -2.5606897, 4.9691135, 12.623043, 2.7801199, -0.62472878, 1.0660781,
        -1.1269195, -1.5641197, 1.0650766, 0.39323638, 40.935585, -2.1769484, 4.8806390, 4.0958495,
        19.533863, 6.5828492, 12.716765, 0.62841618, -2.7156363, 0.37483101, -2.1719868, 0.24281508,
        -1.4238354, 0.53363149, 3.3997936, 4.3179975, 3.0070405, 1.9244954, 0.97303874, 0.16622487,
        56.249846, 0.99759795, -9.0170971, -0.0022892834, 2.4332286, 13.505042, 2.7273094, 0.85286147,
        6.2476052, 1.1668279, 21.425502, -3.5858604, 11.366853, 4.4198128, 23.833542, 31.437872,
        12.800404, 1.5921705, 7.2102793, 2.0118020, -13.326060, 13.516557, -5.5521307, -7.6348338,
        -43.517188, -30.914650, -26.136892, -1.5207673, -4.0960436, 0.39359664, -50.347672, 8.6261305,
        -3.6101806, -3.0502790, -15.744930, 4.7731851, -10.537163, 4.0519305, 7.2375283, 0.24705012,
        -10.015663, -4.9107503, 17.049900, 5.0112081, -13.155088, -39.355624, -13.154626, -4.9301854,
        -20.182232, 1.4104714, -18.380964, -4.9058427, -6.3258538, 0.14983791, 5.4076778, 0.81244368,
        0.38398144, 4.1942780, -3.2145544, 0.28927988, 73.923797, -15.147222, -8.8879035, 2.6011473,
        44.018005, 31.574038, 35.293598, 3.4923259, 1.1570867, -2.2002833, -44.188137, 5.3891850,
        3.5639276, -0.49887963, -19.871951, -14.308017, -13.614183, -1.3195984, 2.0342390, -0.35502973,
        -67.697354, 9.4404628, 18.362139, 1.0058470, -27.102740, -11.391561, -21.357225, -2.7836197,
        2.6675338, -0.53227508, 52.798195, -4.6978395, -11.966006, -2.6336658, 8.5526304, -3.3786951,
        6.4437157, -4.3903650, -2.9090069, -0.90292220, -51.538154, 1.7151844, 4.3185658, -3.8777125,
        -15.329524, -6.4979235, -12.819149, -8.1285503, 4.9989575, -6.1421902, -8.9984145, 0.27577879,
        -2.8189280, -4.6639381, -14.429005, -0.039993961, 3.8626173, -7.4937793, 5.7972079, -7.6863202,
        -37.592792, -6.2852011, 7.3073436, 3.8185854, 7.4141533, -2.4605615, 2.2687465, 0.91067867,
        5.0105457, 3.8264403, -18.693893, -4.0241736, -4.8465493, -1.4068423, -3.9087303, -8.4383115,
        0.60251699, 0.80978245, 5.0777901, 0.67118076, -30.341660, -0.21426767, 7.3199788, -1.3941479,
        -12.654811, -9.2811407, -5.7703032, 0.46057307, 3.1177503, 3.2379333, 14.136453, -2.8220574,
        12.958732, 3.5413501, 16.501701, 1.1323695, -0.80629852, 2.5160334, -4.0292649, 4.3201633,
        -2.6913222, 5.6773483, -73.398314, -21.778012, 3.4835716, 19.225979, 8.8561544, -0.89693700,
        2.9567284, -2.7231488, 4.9163078, 7.9785416, -23.316471, -4.1985871, 19.694066, 17.706385,
        5.8257647, 0.29198249, -0.14646616, 1.1244238, 19.327001, 4.7062149, -24.146253, -9.7951966,
        18.288684, -5.8423771, 3.6263372, 2.5412824, -11.347229, 6.0260470, 14.541660, 2.7592937,
        -27.738786, -9.2360547, 10.107713, 4.0731198, 9.4694512, 1.1507385, -2.6198258, 0.89095188,
        94.338058, 7.9935917, -15.264231, 0.41608016, 26.384307, 7.1399193, 11.056054, 8.7971599,
        -15.077341, 4.8890058, 44.628653, 0.91180419, -1.2572589, 0.15109582, 10.876046, 0.10168736,
        3.0639823, 3.9365562, -13.209234, 4.8321838, 44.402404, 2.2815647, -8.1351925, 3.6496146,
        18.732453, 15.596899, 12.468623, 4.1239651, -5.7539145, -0.18156353, -3.5160004, 2.3564493,
        -11.657260, -1.4399875, -10.262749, -4.2290701, -4.5272052, 3.4277391, -2.7895621, -1.0477540,
        15.971975, -0.81653628, 40.538470, 11.911440, -12.472785, -3.1337690, -5.1231272, -1.3678346,
        0.96612263, -2.2779325, 7.4829803, -3.1057415, 24.358413, 6.6695230, -8.8908910, -4.6806062,
        -8.2125543, 0.0034567928, -2.9490625, -0.93474576, 6.5922005, -5.6909256, 23.711355, 10.454826,
        -11.151205, 5.2714039, -1.5349970, -2.0364751, 8.2713571, -6.0314579, -23.066997, -1.9055690,
        24.809892, 7.4519983, -17.303990, -4.9698872, -8.6692266, -2.7916504, 5.0007343, -2.2756433,
        -10.912519, -9.7835613, 72.566538, 23.908773, -20.388305, -13.698632, -10.901799, -2.6220785,
        6.4224279, -4.2336067, -25.153578, -7.1228482, 37.272253, 9.8549369, -25.831480, -14.359420,
        -11.771693, -2.2791917, 5.5600220, -3.4272112, -31.659134, -8.8201062, 19.030242, 2.7338837,
        -24.644335, -12.060517, -8.5081081, -3.3034870, 11.625792, -0.38816515, -18.187538, -4.6322807,
        29.019947, 8.6686018, 7.6660369, 3.5585597, -3.4752827, 0.39124437, 6.5746802, 4.0903724,
        -9.1816355, 3.2634813, -33.646079, -10.304648, 4.7289115, 9.3336558, 4.7529946, -3.7218258,
        -1.8341768, -3.8550480, -13.279896, 7.6325649, -31.025743, -7.8032047, -1.2848641, 8.9106132,
        2.8055825, -0.55142165, 3.7200418, -4.4192516, 2.5672564, 3.3466923, 10.302698, -0.79022668,
        1.9119924, -11.663306, -5.9006022, -1.5455202, -7.2920701, 1.9717978, 18.660507, 0.22939795,
        -11.639152, -4.7419391, 0.92337001, 1.0312834, 6.2241722, -1.8675722, -2.5994488, -2.2088180,
        5.6729598, -1.9163577, 6.9429642, 2.5975492, 15.154817, -8.6517013, 0.58815928, 5.7410250,
        -4.5560073, 6.0609943, -5.1303529, -3.0157456, -1.4544771, -2.1681495, -2.5465926, -9.9109914,
        4.5132911, 0.034353320, -0.52332625, 2.6309107, -9.6387115, 5.4870403, -37.550227, -11.837462,
        1.2337675, 15.068695, 5.6790059, -0.82180690, 2.3215222, -1.0103627, 15.441083, 3.6796557,
        -12.236272, -1.7153865, 15.169027, 9.7082687, 2.1467077, 1.9135164, -3.2289422, 1.8556578,
        0.013279370, 0.000094441644, 0.00086814291, 0.0011507320, 0.00061436755, 0.028783052,
        -0.0017508923, 0.0015285429, 0.000067371711, 0.00097976664, 0.020882977, -0.00032760591,
        0.0012917751, 0.000065106370, 0.00022257649, 0.038003076, -0.0013947321, 0.0022317608,
        0.0012923790, 0.0020669744
    ])

    psi_low, psi_high = -0.6, 0.6
    bzimf_low, bzimf_high = -7.0, 7.0
    t_low, t_high = 0.13000, 0.300000
    dt_low, dt_high = 0.1100000, 0.2500000

    fps = (psi * 2.0 - psi_high - psi_low) / (psi_high - psi_low)
    fps2 = fps ** 2
    fbz = (bzimf * 2.0 - bzimf_high - bzimf_low) / (bzimf_high - bzimf_low)
    fbz21 = np.abs(fbz)
    fth = (th * 2.0 - t_high - t_low) / (t_high - t_low)
    fdth = (dth * 2.0 - dt_high - dt_low) / (dt_high - dt_low)

    a0 = a[nlin] + a[nlin + 1] * fbz + a[nlin + 2] * fbz21 + a[nlin + 3] * fth + a[nlin + 4] * fdth
    b0 = a[nlin + 5] + a[nlin + 6] * fbz + a[nlin + 7] * fbz21 + a[nlin + 8] * fth + a[nlin + 9] * fdth

    fx = 0.0
    fy = 0.0
    fz = 0.0

    ind = -20

    for i in range(1, 4):
        for k in range(1, 6):
            a_i = a0 * i
            b_k = b0 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)
            bx = xk * ex * cay * sbz
            by = -a_i * ex * say * sbz
            bz = b_k * ex * cay * cbz

            ind += 20

            ai = (a[ind] + a[ind + 1] * fbz + a[ind + 2] * fth + a[ind + 3] * fdth + a[ind + 4] * fbz21 +
                  a[ind + 5] * fth ** 2 + a[ind + 6] * fdth ** 2 + a[ind + 7] * fbz * fth + a[ind + 8] * fth * fdth +
                  a[ind + 9] * fbz * fdth + a[ind + 10] * fps2 + a[ind + 11] * fbz * fps2 + a[ind + 12] * fth * fps2 +
                  a[ind + 13] * fdth * fps2 + a[ind + 14] * fbz21 * fps2 + a[ind + 15] * fth ** 2 * fps2 +
                  a[ind + 16] * fdth ** 2 * fps2 + a[ind + 17] * fbz * fth * fps2 + a[ind + 18] * fth * fdth * fps2 +
                  a[ind + 19] * fbz * fdth * fps2)

            fx += ai * bx
            fy += ai * by
            fz += ai * bz

    # Parallel component
    a1 = a[nlin + 10] + a[nlin + 11] * fbz + a[nlin + 12] * fbz21 + a[nlin + 13] * fth + a[nlin + 14] * fdth
    b1 = a[nlin + 15] + a[nlin + 16] * fbz + a[nlin + 17] * fbz21 + a[nlin + 18] * fth + a[nlin + 19] * fdth

    for i in range(1, 4):
        for k in range(1, 6):
            a_i = a1 * i
            b_k = b1 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)
            bx = xk * ex * cay * cbz
            by = -a_i * ex * say * cbz
            bz = -b_k * ex * cay * sbz

            ind += 20

            ai = (a[ind] + a[ind + 1] * fbz + a[ind + 2] * fth + a[ind + 3] * fdth + a[ind + 4] * fbz21 +
                  a[ind + 5] * fth ** 2 + a[ind + 6] * fdth ** 2 + a[ind + 7] * fbz * fth + a[ind + 8] * fth * fdth +
            a[ind + 9] * fbz * fdth + a[ind + 10] * fps2 + a[ind + 11] * fbz * fps2 + a[ind + 12] * fth * fps2 +
            a[ind + 13] * fdth * fps2 + a[ind + 14] * fbz21 * fps2 + a[ind + 15] * fth ** 2 * fps2 +
            a[ind + 16] * fdth ** 2 * fps2 + a[ind + 17] * fbz * fth * fps2 + a[ind + 18] * fth * fdth * fps2 +
            a[ind + 19] * fbz * fdth * fps2)

            fx += ai * bx * fps
            fy += ai * by * fps
            fz += ai * bz * fps
    return fx, fy, fz

@jit(fastmath=True, nopython=True)
def r1_fac_shld(x, y, z, psi, bzimf, th, dth):
    nlin = 600
    ntot = 620

    a = np.array([
        -15.271347, 1.6455582, 10.794049, 6.0306370, -3.7796018, -2.8885203, -2.9401536, -0.011517207,
        -0.95736669, 0.14868986, -4.9341796, -0.39183480, 8.3385044, 4.1259764, -0.19308437, 0.75095926,
        -0.66509934, -0.33958269, -0.21049905, 0.54950447, -20.245188, 1.2002129, 8.6501459, 5.0327667,
        -3.9696399, -0.097747152, -2.5714432, 0.34820790, 1.3263432, -0.10239796, -5.8123136, 1.5406234,
        2.1156842, 2.6777544, 0.51059055, 2.1020729, 0.41322944, 1.2779064, 0.49081057, 0.34228743,
        -17.817429, 1.5597554, -5.1901082, -1.2148744, -2.4417820, 5.6969302, -0.12102459, 1.2019761,
        3.9745854, -0.46030737, -0.90364498, 0.060152450, -2.4436798, 0.10173255, 3.4493795, 6.2586263,
        2.3222067, 0.31817308, 1.8608735, -0.10522218, -6.8820843, -0.27314442, -13.906148, -5.2933769,
        1.5731330, 6.7126454, 2.1538384, 1.4827180, 3.9447130, -0.65058900, 1.6468335, -0.35812963,
        -6.9564424, -2.9201997, 3.8314242, 8.0418680, 3.3335992, 0.87916089, 2.6853615, 0.10966281,
        7.2656848, -0.23581093, 0.58898660, 1.0101912, 0.48098264, -6.5340398, -0.48577992, -0.16852617,
        -2.7049354, 0.34960232, -0.58969991, -0.093116760, 6.5205225, 1.3625373, -3.0528431, -2.2322458,
        -2.4510668, -0.88218763, -1.5160863, 0.50509015, -3.7700492, 0.83926659, 0.93972517, 0.90487308,
        -0.56041950, -0.68487029, -0.52442145, -0.080175281, -0.35872245, -0.13644167, -0.74689715, -0.69208022,
        2.3492683, 0.72703004, 0.25468799, 1.1596298, 0.063014265, -0.44609039, 0.35440889, 0.18849753,
        -3.9282029, -0.27465231, 2.5988569, 1.2623058, -0.35005605, -0.19941512, -0.38035664, -0.016283901,
        0.25995450, -0.063761202, -1.4549721, 0.80756960, -1.0589640, 0.10273842, -0.15865484, 0.15646497,
        0.12046765, 0.88456828, 0.027310353, 0.12749888, -3.5400198, 0.52936534, -1.2706701, -0.44412937,
        -0.84449367, 1.4171252, 0.010826550, 0.28098975, 0.88322759, 0.022113253, 0.56657521, -0.29464927,
        0.28566963, 0.31361521, 0.79652763, 0.89884955, 0.21814103, -0.35984978, 0.052874298, -0.10309188,
        -0.98187577, -0.12658854, -2.6852877, -0.94813949, 0.45426445, 1.0947740, 0.35495148, 0.22988848,
        0.65912711, -0.19000311, -0.12128984, 0.062978988, -1.3626784, -0.63676349, 0.64950842, 1.6687608,
        0.70220448, 0.25646153, 0.64548394, 0.028869747, 1.2766155, 0.0080824494, 0.34169410, 0.26469269,
        0.011028794, -1.3307173, -0.14910077, -0.076727833, -0.55949775, 0.051985948, -0.13283266, -0.011194929,
        1.3195809, 0.31139764, -0.63577909, -0.54538963, -0.53904450, -0.20820291, -0.34271554, 0.074115945,
        14.744764, -0.53145884, -14.471161, -7.1582553, 4.4989763, 2.5618190, 3.2795746, -0.14166576,
        0.43619356, -0.54915407, 5.8260177, -1.2283109, -6.9738932, -4.6145226, 0.81796983, 1.5809610,
        1.1281105, -0.58986145, 1.1530497, -0.36273720, 22.759656, -2.7101406, -6.9004168, -4.7322580,
        5.4513842, -0.61918332, 3.1242857, -0.58928693, -1.6025682, 0.037484820, 5.5337346, -0.40620304,
        -5.9935864, -3.9932245, -1.3928743, -3.2375316, -0.45053899, 0.25856228, -0.80998403, -0.20042797,
        19.902912, -1.1681881, 5.4888067, 0.96777133, 1.6895499, -5.7814532, 0.17962567, -1.2088116,
        -4.2593062, 0.84851590, 2.8949595, -0.88022408, 4.9435841, 0.75208268, -3.6514418, -8.0624743,
        -3.2861891, -1.4477105, -2.9595274, -0.077666524, 8.6846728, 0.082202688, 15.852551, 6.2380180,
        -1.3994554, -8.1068881, -2.5738740, -1.7677688, -4.7093335, 0.61532335, -2.9996223, 0.73196267,
        7.8040331, 3.1332764, -4.5546231, -8.8255868, -3.5743152, -0.70465782, -2.6879628, -0.067545690,
        -8.6095912, 0.27881259, -0.72853789, -1.2551575, -0.56442154, 7.8232208, 0.64723331, 0.24016248,
        3.2422148, -0.36691553, 0.96618466, 0.023334320, -7.8388137, -1.6157130, 3.6688026, 2.6166092,
        2.9028760, 1.0456438, 1.7553525, -0.56899779, 28.658100, 1.3748015, -19.699179, -10.363964,
        6.7302157, 4.8852361, 3.9664115, 1.1124231, 0.023591186, -0.038879137, 1.6086627, 1.7900439,
        0.32760757, -0.95989632, -3.2103181, -3.0728502, -2.0623461, -0.41712777, -1.3065570, 0.45390196,
        16.918680, 2.5778391, -7.9210732, -5.5431623, 1.0937879, 1.0697655, -0.44334859, -0.56039369,
        0.16106746, -0.14101467, 0.32124323, 0.34956103, -5.9686606, -2.0614439, -2.4685552, -3.9290120,
        -1.1793542, -0.34293127, -1.9829915, -0.27968113, 17.206384, 0.060588211, -2.9892193, -1.8801090,
        3.2879687, -1.9402996, 0.57042872, -0.53939826, -3.3469087, -0.13274444, -6.2240681, 1.8355204,
        2.1284208, -0.47472016, -3.4904048, -0.16264498, -2.4523527, 0.046724794, 0.22944981, -0.17774443,
        11.058877, -1.5369009, 3.0285393, 0.89442778, -0.35247281, -5.2151919, -0.73866222, -0.019786357,
        -3.5246426, -0.21293137, -8.9984138, -0.52127875, -0.72286490, -0.38615685, -3.6457241, -0.065314502,
        -3.1311006, 0.53364877, 0.96775472, -0.038957320, 7.3142401, -1.4937863, 6.6840662, 3.1210862,
        -2.1008612, -3.0756274, -2.1320448, -0.72635450, -2.4932463, -0.35244573, -2.9446736, -0.45751048,
        1.2736560, 0.72853475, -5.8463491, -6.6366170, -3.4866725, -0.51077400, -1.4522607, 0.012547474,
        -22.271917, -4.3006521, 14.693966, 10.707686, -6.5674574, -3.0274418, -3.0829797, -0.10423623,
        1.6436359, -0.096382116, -1.0802533, -1.1004701, 9.6740785, 5.3088323, 2.2878260, 3.3496046,
        1.3450454, 0.62216024, 0.64914279, -0.063041422, -22.569474, -2.1015809, 9.8015664, 6.3442363,
        -5.6177933, 0.41512994, -1.9005245, 0.038357375, 3.9667645, 0.36235727, 3.9240307, -2.8495313,
        0.31571409, 1.9687928, 4.9592455, 2.0991413, 3.7139562, -0.13699646, -0.40288973, -0.037876500,
        -17.246650, -0.63639505, 0.81774006, 2.1439765, -1.4099567, 1.8831939, 0.46124140, 0.75292234,
        2.6316429, 0.46756760, 2.6403981, 0.36476495, 4.7251422, 1.6857628, 4.3796450, 4.6533856,
        3.1774415, 0.002709773, 1.1402869, 0.21424338, -16.129484, 1.1223483, -2.4800571, -0.80199872,
        -0.31533894, 2.9684968, 1.3154297, 0.63822906, 3.5418912, 0.34736259, 5.6013686, -0.39980334,
        -2.8014159, -0.44095632, 7.0921026, 6.4916702, 4.5484625, 0.56539747, 1.4776117, -0.041901881,
        -13.257860, 2.9923037, -9.2499795, -4.5923265, 3.2409752, 6.8435096, 2.9923162, 0.69521579,
        4.5134991, 0.40443866, 10.292041, 0.64830330, -3.2106312, -1.3201682, 8.6475305, 6.0254521,
        5.6590543, 0.12529398, 0.84212776, 0.098205483, 10.514875, 2.9950672, -1.6466923, -2.7019114,
        2.0803189, -3.3957506, 0.60163500, -0.97476044, -3.8814902, -0.043541934, -4.7895029, 2.1570065,
        -4.1647884, -2.8612442, -0.66695727, 2.6819667, -1.4084441, 1.1161515, 1.7538392, -0.46419258,
        4.5374642, -0.37759033, 0.49917414, -0.95048593, 1.6217139, 0.82512846, 1.1817115, 0.14621015,
        -0.18874313, -0.12607646, 1.1376532, -2.0373894, -4.4027318, -1.6545852, 0.26318020, -2.1844168,
        -0.17155302, -0.56049358, -0.42054812, 0.15966903, 2.8251980, 0.053641612, -2.9961682, -1.3370245,
        0.48834767, 0.91620855, -0.16793073, 0.63239697, -0.52617606, -0.19742096, 0.41812179, 0.99606778,
        4.1724508, 0.77979356, -3.7570530, -4.6878614, -2.2182782, -0.78568775, -0.78714011, 0.47883871,
        4.6188524, 0.43323190, 2.2791890, 0.51306074, -0.066258366, 0.50118903, -0.91648888, -1.0542866,
        -0.28370795, -0.20909786, 3.3100384, -0.040019597, -2.1903989, -0.18467609, -2.1639749, -4.1573028,
        -0.91172148, -0.15371192, -1.7381971, -0.28526189, 5.8237338, -1.4409654, 2.1910925, 1.4120539,
        -1.0705890, -3.3078473, -0.82434613, 0.086620485, -1.9265913, -0.050649182, -6.9711352, -0.11068921,
        2.8955611, 0.73126502, -3.0110206, -0.20011953, -2.2363997, 0.13554367, 0.35218065, -0.026202084,
        0.0093470292, 0.00046516521, 0.0051572626, -0.012140839, 0.00013652830, 0.067482845, -0.043183269,
        0.010169675, 0.00014055113, 0.0087104557, 0.026922377, -0.010044979, 0.014457557, 0.00060191484,
        0.0020030915, 0.054593113, -0.014994350, 0.025290933, 0.023759393, 0.017313238
    ])

    psi_low, psi_high = -0.6, 0.6
    bzimf_low, bzimf_high = -7.0, 7.0
    t_low, t_high = 0.13000, 0.300000
    dt_low, dt_high = 0.1100000, 0.2500000

    fps = (psi * 2.0 - psi_high - psi_low) / (psi_high - psi_low)
    fps2 = fps ** 2
    fbz = (bzimf * 2.0 - bzimf_high - bzimf_low) / (bzimf_high - bzimf_low)
    fbz21 = np.abs(fbz)
    fth = (th * 2.0 - t_high - t_low) / (t_high - t_low)
    fdth = (dth * 2.0 - dt_high - dt_low) / (dt_high - dt_low)

    a0 = a[nlin] + a[nlin + 1] * fbz + a[nlin + 2] * fbz21 + a[nlin + 3] * fth + a[nlin + 4] * fdth
    b0 = a[nlin + 5] + a[nlin + 6] * fbz + a[nlin + 7] * fbz21 + a[nlin + 8] * fth + a[nlin + 9] * fdth

    fx = 0.0
    fy = 0.0
    fz = 0.0

    ind = -20

    for i in range(1, 4):
        for k in range(1, 6):
            a_i = a0 * i
            b_k = b0 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)
            bx = xk * ex * cay * sbz
            by = -a_i * ex * say * sbz
            bz = b_k * ex * cay * cbz

            ind += 20

            ai = (a[ind] + a[ind + 1] * fbz + a[ind + 2] * fth + a[ind + 3] * fdth + a[ind + 4] * fbz21 +
                  a[ind + 5] * fth ** 2 + a[ind + 6] * fdth ** 2 + a[ind + 7] * fbz * fth + a[ind + 8] * fth * fdth +
                  a[ind + 9] * fbz * fdth + a[ind + 10] * fps2 + a[ind + 11] * fbz * fps2 + a[ind + 12] * fth * fps2 +
                  a[ind + 13] * fdth * fps2 + a[ind + 14] * fbz21 * fps2 + a[ind + 15] * fth ** 2 * fps2 +
                  a[ind + 16] * fdth ** 2 * fps2 + a[ind + 17] * fbz * fth * fps2 + a[ind + 18] * fth * fdth * fps2 +
                  a[ind + 19] * fbz * fdth * fps2)

            fx += ai * bx * fps
            fy += ai * by * fps
            fz += ai * bz * fps

    # Parallel component
    a1 = a[nlin + 10] + a[nlin + 11] * fbz + a[nlin + 12] * fbz21 + a[nlin + 13] * fth + a[nlin + 14] * fdth
    b1 = a[nlin + 15] + a[nlin + 16] * fbz + a[nlin + 17] * fbz21 + a[nlin + 18] * fth + a[nlin + 19] * fdth

    for i in range(1, 4):
        for k in range(1, 6):
            a_i = a1 * i
            b_k = b1 * k
            xk = np.sqrt(a_i ** 2 + b_k ** 2)
            ex = np.exp(xk * x)
            cay = np.cos(a_i * y)
            say = np.sin(a_i * y)
            cbz = np.cos(b_k * z)
            sbz = np.sin(b_k * z)
            bx = xk * ex * cay * cbz
            by = -a_i * ex * say * cbz
            bz = -b_k * ex * cay * sbz

            ind += 20

            ai = (a[ind] + a[ind + 1] * fbz + a[ind + 2] * fth + a[ind + 3] * fdth + a[ind + 4] * fbz21 +
                  a[ind + 5] * fth ** 2 + a[ind + 6] * fdth ** 2 + a[ind + 7] * fbz * fth + a[ind + 8] * fth * fdth +
                  a[ind + 9] * fbz * fdth + a[ind + 10] * fps2 + a[ind + 11] * fbz * fps2 + a[ind + 12] * fth * fps2 +
                  a[ind + 13] * fdth * fps2 + a[ind + 14] * fbz21 * fps2 + a[ind + 15] * fth ** 2 * fps2 +
                  a[ind + 16] * fdth ** 2 * fps2 + a[ind + 17] * fbz * fth * fps2 + a[ind + 18] * fth * fdth * fps2 +
                  a[ind + 19] * fbz * fdth * fps2)

            fx += ai * bx
            fy += ai * by
            fz += ai * bz

    return fx, fy, fz

@jit(fastmath=True, nopython=True)
def dipole(ps, x, y, z):
    sps = np.sin(ps)
    cps = np.cos(ps)

    p = x ** 2
    u = z ** 2
    v = 3.0 * z * x
    t = y ** 2
    q = 30574.0 / np.sqrt(p + t + u) ** 5
    bx = q * ((t + u - 2.0 * p) * sps - v * cps)
    by = -3.0 * y * q * (x * sps + z * cps)
    bz = q * ((p + t - 2.0 * u) * cps - v * sps)

    return bx, by, bz




