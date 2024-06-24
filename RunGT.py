import os
import argparse

import numpy as np
from datetime import datetime

from GT.functions import PlotTracks
from GT import Regions, Units
from GT.Algos import BunemanBorisSimulator

parser = argparse.ArgumentParser()
parser.add_argument("--folder")
parser.add_argument("--seed", type=int)
parser.add_argument("--R", type=float, default=1.0)

args = parser.parse_args()
folder = args.folder
seed = args.seed
R = args.R

np.random.seed(seed)


Region = Regions.Heliosphere
# Bfield = "Dipole"
# Bfield = ["Gauss", {'model': "CHAOS", "model_type": "core", "version": 7.13}]
# Bfield = "Parker"
Bfield = ["Parker", {"use_noise": False, "noise_num": 1024, "log_kmax": 6, "use_reg": True, "coeff2d": 0.5}]
Date = datetime(2008, 1, 1)

Flux = ["Uniform",
        {"MinT": 0.1*1e3, "MaxT": 20*1e3, "Center": np.array([0, 0, 0]), "Radius": 30, "Nevents": 100}]
# Flux = ["PowerSpectrum", {"EnergyMin": 0.1, "EnergyMax": 0.5, "RangeUnits": 'T', "Base": 'R', "SpectrumIndex":
# -2.7, "Radius": 5, "Nevents": 5}]
# Flux = "PowerSpectrum"
# Flux = "Monolines"

Nfiles = 200
Output = f"{folder}" + os.sep + "Uniform0.1_20"
# Output = None
Save = [100, {"Angles": True}]
# Save = [10, {"Clock": False, "Path": False, "Bfield": True, "Efield": True, "Energy": True, "Angles": False}]

Verbose = True

# BreakConditions = None
# BreakConditions = {"Xmin": 0, "Ymin": 0, "Zmin": 0, "Rmin": 0, "Dist2Path": 0,
#                    "Xmax": np.inf, "Ymax": np.inf, "Zmax": np.inf, "Rmax": np.inf, "MaxPath": np.inf,
#                    "MaxTime": np.inf}
# BreakConditions = {"Rmax": 40}
BreakConditions = None
simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Particles=Flux, Num=int(1e6), Step=1e-2,
                                  Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose,
                                  BreakCondition=BreakConditions)
simulator()

# tracks, wout = simulator()

# PlotTracks(tracks, simulator.Bfield.Units)
