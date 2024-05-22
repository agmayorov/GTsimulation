import os

import numpy as np
from datetime import datetime

from GT.functions import PlotTracks
from GT import Regions, Units
from GT.Algos import BunemanBorisSimulator

Region = Regions.Magnetosphere
# Bfield = "Dipole"
Bfield = ["Gauss", {'model': "IGRF", "model_type": "core", "version": 13, "coord": 1}]
# Bfield = ["Parker", {"use_noise": True}]
Date = datetime(2006, 6, 15)

Flux = ["Monolines",
        {"T": 100, "Center": np.array([1.5, 0, 0]) * Units.RE2km, "Radius": 0, "V0": np.array([1, 1, 1]), "Nevents": 1}]
# Flux = ["PowerSpectrum", {"EnergyMin": 0.1, "EnergyMax": 0.5, "RangeUnits": 'T', "Base": 'R', "SpectrumIndex":
# -2.7, "Radius": 5, "Nevents": 5}]
# Flux = "PowerSpectrum"
# Flux = "Monolines"

Nfiles = 1
Output = "IGRFtest" + os.sep + "IGRFtest1000"
Save = [1, {"Path": True}]
# Save = [10, {"Clock": False, "Path": False, "Bfield": True, "Efield": True, "Energy": True, "Angles": False}]

Verbose = True

# BreakConditions = None
# BreakConditions = {"Xmin": 0, "Ymin": 0, "Zmin": 0, "Rmin": 0, "Dist2Path": 0,
#                    "Xmax": np.inf, "Ymax": np.inf, "Zmax": np.inf, "Rmax": np.inf, "MaxPath": np.inf,
#                    "MaxTime": np.inf}
BreakConditions = {"Rmin": 1 * Units.RE2km, "Rmax": 20 * Units.RE2km, "MaxPath": 1000 * Units.RE2km}
# BreakConditions = None
simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Particles=Flux, Num=int(100000000),
                                  Step=1e-5,
                                  Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose,
                                  BreakCondition=BreakConditions)
simulator()

# tracks, wout = simulator()

# PlotTracks(tracks, simulator.Bfield.Units)
