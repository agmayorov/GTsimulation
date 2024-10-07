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

np.random.seed(seed)

# Region = Regions.Magnetosphere
# Bfield = "Dipole"
# Bfield = ["Gauss", {"model": "CHAOS", "model_type": "core", "version": 7.13}]

Region = [Regions.Heliosphere, {"CalcAdditionalEnergy": True}]
# Region = Regions.Heliosphere
Bfield = "Parker"
# Bfield = ["Parker", {"use_noise": False, "noise_num": 1024, "log_kmax": 6, "use_reg": True, "coeff2d": 0.5}]

# Region = Regions.Galaxy
# Bfield = "JF12mod"

Date = datetime(2008, 1, 1)

Medium = None
# Medium = ["GTnrlmsis", {"version": 0}]

# Flux = {"Distribution": "Disk", "Nevents": 10000, "T": 200, "Radius": 14, "Width": 0.2}
Flux = {"Nevents": 1, "T": 10, "Names": "pr", "Radius": 0, "Center": np.array([5, 5, 0]), "V0": np.array([-0.58762716,  0.79426625, -0.15438733])}

UseDecay = True
NuclearInteraction = None
# NuclearInteraction = {"GenMax": 3}

Nfiles = 1
# Output = None
# Output = "Galaxy"
Output = f"{folder}" + os.sep + "test"
# Save = [1, {"Clock": True, "Path": True, "Density": True}]
Save = [1, {"Energy": True}]
# Save = [10, {"Clock": False, "Path": False, "Bfield": True, "Efield": True, "Energy": True, "Angles": False}]

Verbose = True

BreakConditions = None
# BreakConditions = {"Rmax": 28.5}
# BCcenter = np.array([-8.5, 0, 0])

simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Medium=Medium, Particles=Flux, Num=int(5e6),
                                  Step=0.1, Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose, UseDecay=UseDecay,
                                  InteractNUC=NuclearInteraction, BreakCondition=BreakConditions)
simulator()
