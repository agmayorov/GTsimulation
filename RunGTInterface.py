import os
import argparse

import numpy as np
from datetime import datetime

from Global import Units as U
from Interface import InterfaceGT

parser = argparse.ArgumentParser()
parser.add_argument("--folder")
parser.add_argument("--seed", type=int)
parser.add_argument("--R", type=float, default=1.0)

args = parser.parse_args()
folder = args.folder
seed = args.seed
R = args.R

folder = r"tests/Mag"

np.random.seed(seed)

Region = "Magnetosphere"
# Bfield = "Dipole"
Bfield = ["Gauss", {"model": "CHAOS", "model_type": "core", "version": 7.13}]

# Region = [Regions.Heliosphere, {"CalcAdditionalEnergy": True}]
# Region = Regions.Heliosphere
# Bfield = "Parker"
# Bfield = ["Parker", {"use_noise": False, "noise_num": 1024, "log_kmax": 6, "use_reg": True, "coeff2d": 0.5}]

# Region = Regions.Galaxy
# Bfield = "JF12mod"

Date = datetime(2008, 1, 1)

# Medium = None
Medium = ["GTnrlmsis", {"version": 0}]

# Flux = {"Distribution": "Disk", "Nevents": 10000, "T": 200, "Radius": 14, "Width": 0.2}
Flux = {"Nevents": 1, "T": 20*U.GeV, "Names": "pr", "Radius": 0, "Center": np.array([1.5*U.RE, 0, 0]), "Mode": "Inward"}

UseDecay = False
NuclearInteraction = None
# NuclearInteraction = {"GenMax": 3}

Nfiles = [80]
# Output = None
# Output = "Galaxy"
Output = f"{folder}" + os.sep + "MaxRevTest"
# Save = [1, {"Clock": True, "Path": True, "Density": True}]
# Save = 1
Save = [1, {"Bfield": True, "Clock": True, "Coordinates": False}]

Verbose = True

# BreakConditions = None
BreakConditions = {"Rmax": 80*U.AU, 'MaxRev': 1}
# BCcenter = np.array([-8.5, 0, 0])

tracks = InterfaceGT(Date=Date, Region=Region, Bfield=Bfield, Medium=Medium, Particles=Flux, Num=int(1e3),
                                  Step=1e-6, Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose, UseDecay=UseDecay,
                                  InteractNUC=NuclearInteraction, BreakCondition=BreakConditions, ForwardTrck=1)
