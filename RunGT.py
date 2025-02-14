import os
import argparse

import numpy as np
from datetime import datetime

from Global import Regions
from Global import Units as U
from GT.Algos import BunemanBorisSimulator, RungeKutta4Simulator, RungeKutta6Simulator
from MagneticFields.Magnetosphere import Gauss
from Particle import Flux
from Medium.Magnetosphere import GTnrlmsis

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
Date = datetime(2008, 1, 1)

Region = Regions.Magnetosphere
# Bfield = "Dipole"
Bfield = Gauss(date=Date, model='CHAOS', model_type='core', version=7.13)

# Region = [Regions.Heliosphere, {"CalcAdditionalEnergy": True}]
# Region = Regions.Heliosphere
# Bfield = "Parker"
# Bfield = ["Parker", {"use_noise": False, "noise_num": 1024, "log_kmax": 6, "use_reg": True, "coeff2d": 0.5}]

# Region = Regions.Galaxy
# Bfield = "JF12mod"



# Medium = None
Medium = GTnrlmsis(date=Date, version=0)

# Flux = {"Distribution": "Disk", "Nevents": 10000, "T": 200, "Radius": 14, "Width": 0.2}
Flux = Flux(Names='pr', Radius=0, Center=np.array([1.2*U.RE, 0, 0]), T=20*U.GeV, V0=np.array([-1, 0, 0]), Nevents=20)

UseDecay = False
NuclearInteraction = None
# NuclearInteraction = {"GenMax": 3}

Nfiles = 1
# Output = None
# Output = "Galaxy"
Output = f"{folder}" + os.sep + "test"
# Save = [1, {"Clock": True, "Path": True, "Density": True}]
Save = 1

Verbose = True

BreakConditions = None

simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Medium=Medium, Particles=Flux, Num=int(1e3),
                                  Step=1e-5, Save=Save, Nfiles=Nfiles, Output=Output, Verbose=Verbose, UseDecay=UseDecay,
                                  InteractNUC=NuclearInteraction, BreakCondition=BreakConditions, ForwardTrck=1)
simulator()
