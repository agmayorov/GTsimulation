import os

import numpy as np
from datetime import datetime

from GT.functions import PlotTracks
from MagneticFields import Regions
from GT.Algos import BunemanBorisSimulator

Region = Regions.Heliosphere
Bfield = "Parker"
# Bfield = ["Gauss", {'model': "IGRF", "model_type": "core", "version": 13, "coord": 0}]
Date = datetime(2008, 1, 1)

Flux = ["Uniform", {"MinT": 1e3*0.1, "MaxT": 1e-3*20, "Center": np.array([0, 0, 0]), "Radius": 30, "Nevents": 50}]
# Flux = ["PowerSpectrum", {"EnergyMin": 0.1, "EnergyMax": 0.5, "RangeUnits": 'T', "Base": 'R', "SpectrumIndex":
# -2.7, "Radius": 5, "Nevents": 5}]
# Flux = "PowerSpectrum"
# Flux = "Monolines"

Nfiles = 100000
Output = "ParkerPR" + os.sep + "Uniform0.1_20"
Save = 10
# Save = [10, {"Clock": False, "Path": False, "Bfield": True, "Efield": True, "Energy": True, "Angles": False}]

Verbose = True

# BreakConditions = None
# BreakConditions = {"Xmin": 0, "Ymin": 0, "Zmin": 0, "Rmin": 0, "Dist2Path": 0,
#                    "Xmax": np.inf, "Ymax": np.inf, "Zmax": np.inf, "Rmax": np.inf, "MaxPath": np.inf,
#                    "MaxTime": np.inf}
BreakConditions = {"Rmin": 20, "Rmax": 50}

simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Particles=Flux, Num=int(1e3), Step=1,
                                  Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose,
                                  BreakCondition=BreakConditions)
simulator()

# tracks, wout = simulator()

# PlotTracks(tracks, simulator.Bfield.Units)
