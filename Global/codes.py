import numpy as np

BreakCode = {"Loop": -1, "Xmin": 0, "Ymin": 1, "Zmin": 2, "Rmin": 3, "Dist2Path": 4, "Xmax": 5, "Ymax": 6,
             "Zmax": 7, "Rmax": 8, "MaxPath": 9, "MaxTime": 10, "MaxRev": 11, "Death": -2}
"""
Simulation break reasons and the corresponding code.

:param Loop: The simulation went through all its steps

:param Xmin:Ymin:Zmin: The absolute value of *x, y, z* coordinate is less than `Xmin`, `Ymin`, `Zmin` (in :py:mod:`MagneticFields` units)

:param Xmax:Ymax:Zmax: The absolute value of *x, y, z* coordinate is greater than `Xmax`, `Ymax`, `Zmax` (in :py:mod:`MagneticFields` units)

:param Rmin:Rmax: The radius is less(greater) than `Rmin`(`Rmax`) (in :py:mod:`MagneticField` units)

:param Dist2Path: The fraction of the distance travelled to the path is less than the `Dist2Path`.

:param Death: The particle went into an nuclear interaction.

:param MaxPath:MaxTime: The path (time) travelled is greater than the parameters (in :py:mod:`MagneticFields` units for `MaxPath`)

:param MaxRev: The maximum number of revolutions.
"""
BreakIndex = {-1: "Loop", 0: "Xmin", 1: "Ymin", 2: "Zmin", 3: "Rmin", 4: "Dist2Path", 5: "Xmax",
                             6: "Ymax", 7: "Zmax", 8: "Rmax", 9: "MaxPath", 10: "MaxTime", 11: "MaxRev", -2: "Death"}
"""The inverse `dict` to :ref:`BreakCode`"""
BreakDef = np.array([0, 0, 0, 0, 0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
BreakMetric = np.s_[:-1]

SaveCode = {"Coordinates": (1, np.s_[0:3:1]),
            "Velocities": (2, np.s_[3:6:1]),
            "Efield": (3, np.s_[6:9:1]),
            "Bfield": (4, np.s_[9:12:1]),
            "Angles": (5, 12),
            "Path": (6, 13),
            "Density": (7, 14),
            "Clock": (8, 15),
            "Energy": (9, 16)}
"""
The parameters that can be saved along the path and the corresponding indices in the matrix.
"""
SaveMetric = np.array(3*[True]+10*[False]+[True]+3*[False])
SaveDef = {"Coordinates": True, "Velocities": True}
SaveDef.update(dict(zip([key for key in SaveCode.keys() if key != "Coordinates" and key != "Velocities"],
                   [False] * (len(SaveCode.keys())-2))))
