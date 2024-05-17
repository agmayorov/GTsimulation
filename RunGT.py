import numpy as np
from datetime import datetime

from GT.functions import PlotTracks
from MagneticFields import Regions
from GT.Algos import BunemanBorisSimulator

Region = Regions.Magnetosphere
Bfield = ["Dipole", {"M": 30000.0}]
# Bfield = ["Gauss", {'model': "IGRF", "model_type": "core", "version": 13, "coord": 0}]
Date = datetime(2006, 6, 15)

Flux = ["Monolines", {"T": 30, "Center": np.array([1.5, 0, 0]), "Radius": 0, "Nevents": 1, "V0": [-1, 0, 1]}]
# Flux = ["PowerSpectrum", {"EnergyMin": 0.1, "EnergyMax": 0.5, "RangeUnits": 'T', "Base": 'R', "SpectrumIndex":
# -2.7, "Radius": 5, "Nevents": 5}]
# Flux = "PowerSpectrum"
# Flux = "Monolines"

Nfiles = 1
Output = "test_Dip30MeV"
Save = [1, {"Path": True}]
# Save = [10, {"Clock": False, "Path": False, "Bfield": True, "Efield": True, "Energy": True, "Angles": False}]

Verbose = True

BreakConditions = None
# BreakConditions = {"Xmin": 0, "Ymin": 0, "Zmin": 0, "Rmin": 0, "Dist2Path": 0,
#                    "Xmax": np.inf, "Ymax": np.inf, "Zmax": np.inf, "Rmax": np.inf, "MaxPath": np.inf,
#                    "MaxTime": np.inf}
# BreakConditions = {"Rmin": 6378.1371, "Rmax": 6 * 6378.1371}

simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Particles=Flux, Num=int(1e5), Step=1e-4,
                                  Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose,
                                  BreakCondition=BreakConditions)
simulator()

# tracks, wout = simulator()

# PlotTracks(tracks, simulator.Bfield.Units)
