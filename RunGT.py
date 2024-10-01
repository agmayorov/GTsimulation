import os
import argparse

import numpy as np
from datetime import datetime

from Global import Regions, Units
from GT.Algos import BunemanBorisSimulator

parser = argparse.ArgumentParser()
parser.add_argument("--folder")
parser.add_argument("--seed", type=int)
parser.add_argument("--R", type=float, default=1.0)

args = parser.parse_args()
folder = args.folder
seed = args.seed
R = args.R

np.random.seed(seed+189)


# Region = Regions.Galaxy
Region = Regions.Magnetosphere
Bfield = "Dipole"
# Bfield = ["Gauss", {'model': "CHAOS", "model_type": "core", "version": 7.13}]
# Bfield = "Parker"
# Bfield = ["Parker", {"use_noise": False, "noise_num": 1024, "log_kmax": 6, "use_reg": True, "coeff2d": 0.5}]
# Bfield = "JF12mod"
Date = datetime(2008, 1, 1)

Medium = None
# Medium = 'GTnrlmsis'


Flux = {"Distribution": "Disk", "Nevents": 10000, "T": 200, "Radius": 14, "Width": 0.2}
UseDecay = True
NuclearInteraction = None

Nfiles = 1
# Output = f"{folder}" + os.sep + "Uniform0.1_20"
Output = "Galaxy"
# Output = None
# Save = [100, {"Angles": True}]
Save = [0, {"Clock": True}]
# Save = [10, {"Clock": False, "Path": False, "Bfield": True, "Efield": True, "Energy": True, "Angles": False}]

Verbose = True

# BreakConditions = None
# BreakConditions = {"Xmin": 0, "Ymin": 0, "Zmin": 0, "Rmin": 0, "Dist2Path": 0,
#                    "Xmax": np.inf, "Ymax": np.inf, "Zmax": np.inf, "Rmax": np.inf, "MaxPath": np.inf,
#                    "MaxTime": np.inf}
# BreakConditions = {"Rmax": 28.5}
BCcenter = np.array([-8.5, 0, 0])
BreakConditions = None
simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Medium=Medium, Particles=Flux, Num=int(1e2), Step=0.1,
                                  Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose,
                                  BreakCondition=BreakConditions, BCcenter=BCcenter, UseDecay=UseDecay,
                                  InteractNUC=NuclearInteraction)
simulator()

# tracks, wout = simulator()

# PlotTracks(tracks, simulator.Bfield.Units)
