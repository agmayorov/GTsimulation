import numpy as np

BreakCode = {"Loop": -1, "Xmin": 0, "Ymin": 1, "Zmin": 2, "Rmin": 3, "Dist2Path": 4, "Xmax": 5, "Ymax": 6,
             "Zmax": 7, "Rmax": 8, "MaxPath": 9, "MaxTime": 10, "MaxRev": 11, "Death": -2}
BreakIndex = {-1: "Loop", 0: "Xmin", 1: "Ymin", 2: "Zmin", 3: "Rmin", 4: "Dist2Path", 5: "Xmax",
                             6: "Ymax", 7: "Zmax", 8: "Rmax", 9: "MaxPath", 10: "MaxTime", 11: "MaxRev", -2: "Death"}
BreakDef = np.array([0, 0, 0, 0, 0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
BreakMetric = np.s_[:-1]

SaveCode = {"Coordinates": np.s_[:3],
            "Velocities": np.s_[3:6],
            "Efield": np.s_[6:9],
            "Bfield": np.s_[9:12],
            "Angles": 12,
            "Path": 13,
            "Density": 14,
            "Clock": 15,
            "Energy": 16}
SaveMetric = np.array(3*[True]+10*[False]+[True]+3*[False])
SaveDef = dict(zip([key for key in SaveCode.keys() if key != "Coordinates" and key != "Velocities"],
                   [False] * (len(SaveCode.keys())-2)))
