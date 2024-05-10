import numpy as np
from datetime import datetime

from GT.functions import PlotTracks
from MagneticFields import Regions
from GT.Algos import BunemanBorisSimulator

Region = Regions.Magnetosphere
Bfield = ["Dipole", {"M": 30000.0}]
# Bfield = ["Gauss", {'model': "IGRF", "model_type": "core", "version": 13}]
Date = datetime(2006, 7, 5, 12, 00, 00, 0)

Flux = ["Monolines", {"T": 30, "Center": np.array([1.5, 0, 0]), "Radius": 0, "Nevents": 2, "V0": [-1, 0, 1]}]
# Flux = ["PowerSpectrum", {"EnergyMin": 0.1, "EnergyMax": 0.5, "RangeUnits": 'T', "Base": 'R', "SpectrumIndex":
# -2.7, "Radius": 5, "Nevents": 5}]
# Flux = "PowerSpectrum"
# Flux = "Monolines"

Nfiles = 2
Output = None
Save = 1
# Save = [10, {"Clock": False, "Path": False, "Bfield": True, "Efield": True, "Energy": True, "Angles": False}]

Verbose = True

BreakConditions = None
# BreakConditions = {"Xmin": 0, "Ymin": 0, "Zmin": 0, "Rmin": 0, "Dist2Path": 0,
#                    "Xmax": np.inf, "Ymax": np.inf, "Zmax": np.inf, "Rmax": np.inf, "MaxPath": np.inf,
#                    "MaxTime": np.inf}

simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Particles=Flux, Num=int(1e3), Step=1e-4,
                                  Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose,
                                  BreakCondition=BreakConditions)

tracks, wout = simulator()

PlotTracks(tracks, simulator.Bfield.Units)
