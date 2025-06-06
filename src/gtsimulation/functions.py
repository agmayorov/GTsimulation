def GetLastPoints(RetArr_i, s):
    R = RetArr_i["Track"]['Coordinates'][-1]
    V = RetArr_i["Track"]["Velocities"][-1]
    if s == -1:
        V *= -1
    return R, V
