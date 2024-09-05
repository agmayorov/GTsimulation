import numpy as np
import copy
from Global.consts import Units, Constants, Origins
from MagneticFields.Magnetosphere.Functions import transformations
from GT.functions import GetLastPoints
from numba import jit

@jit(fastmath=True, nopython=True)
def AddLon(lon_total, lon_prev, full_revolutions, index, a_, b_):
    lon = np.arctan(b_ / a_)
    lon_tolerance = np.deg2rad(60)

    if index == 0:
        lon_prev = lon

    lon_diff = np.abs(lon - lon_prev)
    if lon_diff > np.pi / 2:
        lon_diff = np.pi - lon_diff

    if lon_diff > lon_tolerance:
        lon_total += lon_diff
        lon_prev = lon

    lon_total = np.abs(lon_total)

    if lon_total > 2 * np.pi:
        full_revolutions += 1
        lon_total = np.remainder(lon_total, 2 * np.pi)

    return lon_total, lon_prev, full_revolutions

def FieldLine(simulator, Rinp, sgn):
    bline = np.array([[0, 0, 0]])
    rline = np.array([Rinp])
    s = 0

    scale = 1e3 * 6370e3 / np.linalg.norm(Rinp)

    while np.linalg.norm(rline[s, :]) > Units.RE2m:
        B1, B2, B3 = simulator.Bfield.GetBfield(rline[s, 0], rline[s, 1], rline[s, 2])
        B = np.sqrt(B1 ** 2 + B2 ** 2 + B3 ** 2)
        d1 = sgn * B1 / B
        d2 = sgn * B2 / B
        d3 = sgn * B3 / B

        bline = np.append(bline, np.array([[B1, B2, B3]]), axis=0)
        rline = np.append(rline, rline[s, :] + Units.RE2m * np.array([[d1, d2, d3]]) / scale, axis=0)
        s += 1

    return rline[1:], bline[1:]

def GetEarthBfieldLine(simulator, rinp):
    urline, ubline = FieldLine(simulator, rinp, 1)
    drline, dbline = FieldLine(simulator, rinp, -1)

    rline = np.concatenate((urline.T, drline.T), axis=1).T
    bline = np.concatenate((ubline.T, dbline.T), axis=1).T

    return rline, bline

def GetTrackParams(Simulator, RetArr_i):
    R = RetArr_i["Track"]["Coordinates"] * Units.RE2m
    H = RetArr_i["Track"]["Bfield"][1:, :]
    M = RetArr_i["Particle"]["M"] * Units.MeV2kg
    T0 = RetArr_i["Particle"]["T0"]
    Z = RetArr_i["Particle"]["Ze"]

    V = (R[1:, :] - R[:-1, :]) / Simulator.Step
    Vn = np.linalg.norm(V, axis=1)
    R = R[1:, :]
    Hn = np.linalg.norm(H, axis=1)
    Y = 1 / np.sqrt((1 - Vn ** 2 / Constants.c ** 2))
    VdotH = np.sum(V * H, axis=1)

    # First invariant
    I1 = M * Y ** 2 * (Vn ** 2 - (VdotH / Hn) ** 2) / (2 * Hn)

    VndotHn = Vn * Hn

    # Pitch angles
    pitch = np.arccos(VdotH / VndotHn) / np.pi * 180

    # Mirror points
    a = pitch[1:] - 90
    a = np.where((pitch[:-1] - 90) * a < 0)[0]
    pitch_bound_tol = 0.4
    n = 0

    i = 0
    while i < a.size - 1:
        if np.max(np.abs(pitch[a[n]:a[n + 1]] - 90)) < pitch_bound_tol:
            a = np.delete(a, n + 1)
        else:
            n += 1
        i += 1

    num_mirror = None
    num_eq_pitch = None
    Hm = None
    Heq = None
    pitch_eq = None
    num_B0 = None

    if a.size > 0:
        num_mirror = np.zeros(a.size, dtype=int)
        num_eq_pitch = np.zeros(a.size - 1, dtype=int)
        num_B0 = np.zeros(a.size - 1, dtype=int)

        num_eq_1 = 0
        for i in range(a.size - 1):
            b = np.argmax(np.abs(pitch[a[i]:a[i + 1]] - 90))
            num_eq_2 = a[i] + b
            d = np.argmax(Hn[num_eq_1:num_eq_2 + 1])
            num_mirror[i] = num_eq_1 + d
            num_eq_1 = num_eq_2
            num_eq_pitch[i] = num_eq_2

            min_index = np.argmin(Hn[a[i]:a[i + 1] + 1])
            num_B0[i] = a[i] + min_index

        d = np.argmax(Hn[num_eq_1:])
        num_mirror[-1] = num_eq_1 + d
        if num_mirror.size == 0:
            num_mirror = None
        else:
            num_mirror = np.unique(num_mirror)
            Hm = Hn[num_mirror]

        if num_eq_pitch.size == 0:
            num_eq_pitch = None
            Heq = None
            pitch_eq = None
        else:
            Heq = Hn[num_eq_pitch]
            pitch_eq = pitch[num_eq_pitch]

        if num_B0.size == 0:
            num_B0 = None

    NumMirror = {"NumMirr": num_mirror, "NumEqPitch": num_eq_pitch, "NumBo": num_B0, "Hmirr": Hm, "Heq": Heq}

    # Second invariant
    I2 = None
    if num_mirror is not None:
        I2 = np.zeros(num_mirror.size - 1)
        I2_tol = 0.2

        for i in range(num_mirror.size - 1):
            HmTmp = max(Hm[i], Hm[i + 1])
            H_coil = Hn[num_mirror[i]:num_mirror[i + 1]]
            b = np.array([R[:, 0][num_mirror[i] + 1:num_mirror[i + 1] + 1],
                          R[:, 1][num_mirror[i] + 1:num_mirror[i + 1] + 1],
                          R[:, 2][num_mirror[i] + 1:num_mirror[i + 1] + 1]])
            S = b - np.array([R[:, 0][num_mirror[i]:num_mirror[i + 1]],
                              R[:, 1][num_mirror[i]:num_mirror[i + 1]],
                              R[:, 2][num_mirror[i]:num_mirror[i + 1]]])

            I2[i] = np.sum(
                np.sqrt(1 - H_coil / HmTmp) * np.abs((S[0, :] * H[:, 0][num_mirror[i]:num_mirror[i + 1]] +
                                                      S[1, :] * H[:, 1][num_mirror[i]:num_mirror[i + 1]] +
                                                      S[2, :] * H[:, 2][num_mirror[i]:num_mirror[i + 1]]) /
                                                     H_coil))

        if I2.size == 0:
            I2 = None
        else:
            I2 = I2[I2 > I2_tol * np.max(I2)]

    if I2 is not None and I2.size == 0:
        I2 = None

    # L-shell and Guiding Centre
    RE = Units.RE2m
    k0 = 31100 * 1e-5  # Gauss * RE^3
    k0 *= 1e-4 * RE ** 3  # Tesla * m^3

    a_new = np.array([
        [3.0062102e-1, 6.2337691e-1, 6.228644e-1, 6.222355e-1, 2.0007187, -3.0460681],
        [3.33338e-1, 4.3432642e-1, 4.3352788e-1, 4.3510529e-1, -1.8461796e-1, 1],
        [0, 1.5017245e-2, 1.4492441e-2, 1.2817956e-2, 1.2038224e-1, 0],
        [0, 1.3714667e-3, 1.1784234e-3, 2.1680398e-3, -6.7310339e-3, 0],
        [0, 8.2711096e-5, 3.8379917e-5, -3.2077032e-4, 2.170224e-4, 0],
        [0, 3.2916354e-6, -3.3408822e-6, 7.9451313e-5, -3.8049276e-6, 0],
        [0, 8.1048663e-8, -5.3977642e-7, -1.2531932e-5, 2.8212095e-8, 0],
        [0, 1.0066362e-9, -2.1997983e-8, 9.9766148e-7, 0, 0],
        [0, 8.3232531e-13, 2.3028767e-9, -3.958306e-8, 0, 0],
        [0, -8.1537735e-14, 2.6047023e-10, 6.3271665e-10, 0, 0]
    ])

    L_shell = None
    GuidingCentre = {}
    LR = None
    LRNit = None
    parReq = None
    parBeq = None
    parBBo = None
    Req = None
    Beq = None
    BBo = None
    Rline = None
    Bline = None

    if Simulator.IsFirstRun:
        if I2 is not None:
            I = np.mean(I2)
            Bm = np.mean(Hm)
            X = np.log(I ** 3 * Bm / k0)

            an = (a_new[:, 0] * (X < -22) + a_new[:, 1] * (-22 < X < 3) + a_new[:, 2] * (-3 < X < 3) + a_new[:,
                                                                                                       3] * (
                          3 < X < 11.7)
                  + a_new[:, 4] * (11.7 < X < 23)) + a_new[:, 5] * (X > 23)

            Y = np.sum(an * X ** np.arange(10))
            L_shell = (k0 / RE ** 3 / Bm * (1 + np.exp(Y))) ** (1 / 3)
            M /= Units.MeV2kg

            # Magnetic field line of Guiding Centre
            gamma = (T0 + M) / M
            omega = np.abs(Z) * Constants.e * Hn[0] / (gamma * M * Units.MeV2kg)

            # Larmor Radius
            LR = np.sin(np.deg2rad(pitch[0])) * np.sqrt(1 - 1 / gamma ** 2) * Constants.c / omega
            LRNit = 2 * np.pi * LR / (Vn[0] * Simulator.Step)

            Nit = min(LRNit + 1, len(R))
            Nit = np.floor(np.arange(0, Nit, Nit / 3 - 1)).astype(int)
            Rmin = np.zeros((Nit.size, 3))

            for i in range(Nit.size):
                Rline, Bline = GetEarthBfieldLine(Simulator, R[Nit[i], :])
                Bn = np.linalg.norm(Bline, axis=1)
                e = np.argmin(Bn)
                Rmin[i, :] = Rline[e, :]
                if i == 0:
                    parReq = Rline[e, :]
                    parBeq = Bline[e, :]
                    parBBo = np.linalg.norm(H[0, :]) / np.linalg.norm(parBeq)

            Rline, Bline = GetEarthBfieldLine(Simulator, np.mean(Rmin, axis=0))
            Bn = np.linalg.norm(Bline, axis=1)
            e = np.argmin(Bn)
            Req = Rline[e, :]
            Beq = Bline[e, :]
            BBo = np.linalg.norm(H[0, :]) / np.linalg.norm(Beq)

            parReqNew = transformations.geo2mag_eccentric(parReq[0], parReq[1], parReq[2], 1, Simulator.ParamDict["Date"])
            GuidingCentre["parL"] = np.linalg.norm(parReqNew) / Units.RE2m

            ReqNew = transformations.geo2mag_eccentric(Req[0], Req[1], Req[2], 1, Simulator.ParamDict["Date"])
            GuidingCentre["L"] = np.linalg.norm(ReqNew) / Units.RE2m

            parReq = parReq / Units.RE2m
            Req = Req / Units.RE2m
            Rline = Rline / Units.RE2m

    GuidingCentre = GuidingCentre | {"LR": LR, "LRNit": LRNit, "parReq": parReq, "parBeq": parBeq, "parBBo": parBBo,
                                     "Req": Req, "Beq": Beq, "BBo": BBo, "Rline": Rline, "Bline": Bline}

    TrackParams_i = {"Invariants": {"I1": I1, "I2": I2},
                     "PitchAngles": {"Pitch": pitch, "PitchEq": pitch_eq},
                     "MirrorPoints": NumMirror,
                     "GuidingCentre": GuidingCentre}

    return TrackParams_i

def GetParticleOrigin(TrackParams_i):
    InitEndFlag = TrackParams_i["InitEndFlag"]
    isFullRevolution = TrackParams_i["isFullRevolution"]
    num_mirror = TrackParams_i["Nm"]
    first_invariant_disp = TrackParams_i["I1_disp"]
    second_invariant_disp = TrackParams_i["I2_disp"]

    first_invariant_disp_tol = 0.3
    first_invariant_disp_tol_2 = 0.7
    second_invariant_disp_tol = 0.3
    full_revolution_tol = 5

    if InitEndFlag[0] == 2:
        origin = Origins.Galactic
        return origin
    if InitEndFlag[0] == 1 and InitEndFlag[1] == 2:
        origin = Origins.Albedo
        return origin
    if InitEndFlag[0] == 3 and InitEndFlag[1] == 2:
        origin = Origins.Unknown
        return origin

    if not isFullRevolution and (InitEndFlag[0] == 3 and InitEndFlag[1] == 3):
        origin = Origins.Unknown
        return origin

    if not isFullRevolution:
        if InitEndFlag[0] == 1 and InitEndFlag[1] == 1:
            if num_mirror is None:
                origin = Origins.Albedo
            if 1 <= num_mirror.size <= 2:
                if first_invariant_disp < first_invariant_disp_tol_2:
                    origin = Origins.Presipitated
                else:
                    origin = Origins.Albedo
            if num_mirror.size > 2:
                if first_invariant_disp < first_invariant_disp_tol_2:
                    origin = Origins.QuasiTrapped
                else:
                    origin = Origins.Albedo
            return origin

        if num_mirror is not None and num_mirror.size <= 2:
            origin = Origins.Unknown
        elif num_mirror is not None and num_mirror.size > 2:
            if first_invariant_disp < first_invariant_disp_tol_2:
                origin = Origins.QuasiTrapped
            else:
                origin = Origins.Albedo
        return origin
    if isFullRevolution:
        if InitEndFlag[0] == 1 or InitEndFlag[1] == 1:
            if first_invariant_disp is not None:
                if first_invariant_disp < first_invariant_disp_tol_2:
                    origin = Origins.Albedo
                else:
                    origin = Origins.QuasiTrapped
            else:
                origin = Origins.QuasiTrapped
            return origin
        if first_invariant_disp is not None and second_invariant_disp is not None:
            if (first_invariant_disp < first_invariant_disp_tol) and (
                    second_invariant_disp < second_invariant_disp_tol) or (isFullRevolution >= full_revolution_tol):
                origin = Origins.Trapped
            else:
                origin = Origins.Unknown
        else:
            origin = Origins.Unknown
    return origin

def GetBCparams(RetArr_i):
    if RetArr_i["BC"]["Status"] == "DefaultBC_Rmin":
        u = 1
    elif RetArr_i["BC"]["Status"] == "DefaultBC_Rmax":
        u = 2
    elif RetArr_i["BC"]["Status"] in ["UserBCfunction", "Done"]:
        u = 3
    else:
        u = 0

    lon_u = RetArr_i["BC"]["lon_total"]
    return u, lon_u

def AddTrajectory(f, b, lonTotal, lon, additions_i, Nm, I1, I2, s):
    InitEndFlag = np.array([f, b])
    lonTotal += lon
    isFullRevolution = 0 + (lonTotal > 2 * np.pi)

    if additions_i['Invariants']['I1'] is not None:
        if s == 1:
            I1 = np.concatenate((I1, additions_i['Invariants']['I1']))
        else:
            I1 = np.concatenate((np.flip(additions_i['Invariants']['I1'][:-1]), I1))

    d = (np.array([np.mean(I1)] * I1.size) - I1) ** 2
    I1_disp = np.sqrt(np.mean(d)) / np.mean(I1)

    # Mirror points
    if additions_i['MirrorPoints']['NumBo'] is not None:
        Nm = np.concatenate((Nm, additions_i['MirrorPoints']['NumBo']))

    # I2
    if additions_i['Invariants']['I2'] is not None:
        if s == 1:
            if additions_i['Invariants']['I2'] is not None:
                I2 = np.concatenate((I2, additions_i['Invariants']['I2']))
        else:
            I2 = np.concatenate((np.flip(additions_i['Invariants']['I2'][:-1]), I2))

    if I2.size == 0:
        I2_disp = None
    else:
        d = (np.array([np.mean(I2)] * I2.size) - I2) ** 2
        I2_disp = np.sqrt(np.mean(d)) / np.mean(I2)

    addTrackDict = {"InitEndFlag": InitEndFlag, "lonTotal": lonTotal, "isFullRevolution": isFullRevolution,
                    "Nm": Nm, "I1": I1, "I1_disp": I1_disp, "I2": I2, "I2_disp": I2_disp}

    return addTrackDict

def FindParticleOrigin(Simulator, RetArr_i):
    # Forward trajectory
    f, lon_f = GetBCparams(RetArr_i)
    addTrackParams = AddTrajectory(f, 0, 0, lon_f, RetArr_i["Additions"], np.array([0]),
                                   np.array([]), np.array([]), 1)
    lon_total = addTrackParams["lonTotal"]
    Nm = addTrackParams["Nm"]
    I1 = addTrackParams["I1"]
    I2 = addTrackParams["I2"]

    Rf, Vf = GetLastPoints(RetArr_i, 1)

    # Backward trajectory
    Date = Simulator.ParamDict["Date"]
    Region = Simulator.ParamDict["Region"]
    Bfield = Simulator.ParamDict["Bfield"]
    Particles = Simulator.ParamDict["Particles"]
    Num = Simulator.ParamDict["Num"]
    Step = Simulator.ParamDict["Step"]
    UseDecay = Simulator.UseDecay
    InteractNUC = Simulator.InteractNUC
    Save = [1, {"Energy": True, "Bfield": True}]
    if isinstance(Particles, list):
        Particles[1]["Nevents"] = 1
    BreakCondition = Simulator.ParamDict["BreakCondition"]
    if BreakCondition is None:
        BreakCondition = {"MaxRev": 5}
    else:
        BreakCondition["MaxRev"] = 5
    bGTsim = copy.deepcopy(Simulator)
    bGTsim.refreshParams(Date=Date, Region=Region, Bfield=Bfield, Particles=Particles, Num=Num, Step=Step, Save=Save,
                         BreakCondition=BreakCondition, TrackParams=True, ParticleOrigin=False, IsFirstRun=False,
                         ForwardTrck=-1, UseDecay=UseDecay, InteractNUC=InteractNUC)

    bGTtrack = bGTsim()
    b, lon_b = GetBCparams(bGTtrack[0][0])

    addTrackParams = AddTrajectory(f, b, lon_total, lon_b, bGTtrack[0][0]["Additions"], Nm, I1, I2, -1)

    Rb, Vb = GetLastPoints(bGTtrack[0][0], -1)

    # Determine the origin of the particle
    origin = GetParticleOrigin(addTrackParams)

    # Repeat procedure
    while origin == Origins.Unknown:
        # Trace extension
        if f == 3:
            s = 1
            if not isinstance(Particles, list):
                Particles = [Particles, {"Center": Rf, "V0": Vf}]
            else:
                Particles[1]["Center"] = Rf
                Particles[1]["V0"] = Vf
            fGTsim = copy.deepcopy(Simulator)
            fGTsim.refreshParams(Date=Date, Region=Region, Bfield=Bfield, Particles=Particles, Num=Num, Step=Step,
                                 Save=Save, BreakCondition=BreakCondition, TrackParams=True, IsFirstRun=False,
                                 UseDecay=UseDecay, InteractNUC=InteractNUC)
            fGTtrack = fGTsim()

            Rf, Vf = GetLastPoints(fGTtrack[0][0], s)
            f, lon = GetBCparams(fGTtrack[0][0])
        else:
            if b == 3:
                s = -1
                if not isinstance(Particles, list):
                    Particles = [Particles, {"Center": Rb, "V0": Vb}]
                else:
                    Particles[1]["Center"] = Rb
                    Particles[1]["V0"] = Vb

                bGTsim.RefreshParams(Date=Date, Region=Region, Bfield=Bfield, Particles=Particles, Num=Num, Save=Save,
                                     Step=Step, BreakCondition=BreakCondition, TrackParams=True, ParticleOrigin=False,
                                     IsFirstRun=False, ForwardTrck=-1, UseDecay=UseDecay, InteractNUC=InteractNUC)
                bGTtrack = bGTsim()

                Rb, Vb = GetLastPoints(bGTtrack[0][0], s)
                b, lon = GetBCparams(bGTtrack[0][0])
            else:
                break

        addTrackParams = AddTrajectory(f, b, lon_total, lon, bGTtrack[0][0]["Additions"], Nm, I1, I2, -1)

        origin = GetParticleOrigin(addTrackParams)

    return origin