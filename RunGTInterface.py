import argparse
import os
from datetime import datetime

import numpy as np

from Global import Units as U
from Interface import InterfaceGT

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)

args = parser.parse_args()
seed = args.seed

np.random.seed(seed)
Date = datetime(2008, 1, 1)

Region = "Magnetosphere"
Bfield = ["Gauss", {"model": "CHAOS", "model_type": "core", "version": 7.13}]

# Medium = None
Medium = ["GTnrlmsis", {"version": 0}]

Particles = {
    "Spectrum": {"Name": "Monolines", "energy": 1.5 * U.GeV},
    "Distribution": {"Name": "SphereSurf", "Radius": 10 * U.RE},
    "Names": "pr",
    "Nevents": 10
}
# Particles = ["FluxPitchPhase", {
#     "Pitch": np.pi / 4,
#     "Phase": np.pi / 6,
#     "Spectrum": {"Name": "Monolines", "energy": 20 * U.GeV},
#     "Distribution": {"Name":  "SphereSurf", "Radius": 0, "Center": np.array([1.2, 0, 0]) * U.RE},
#     "Names": 'pr',
#     "Nevents": 20,
#     "Mode": "Outward"
# }]

UseDecay = False
NuclearInteraction = None
# NuclearInteraction = {"GenMax": 3}

Nfiles = [2]
Output = f"test/filename"
Save = [1, {"Clock": True, "Path": True}]

Verbose = True
BreakConditions = None

tracks = InterfaceGT(Simulator="BunemanBoris",
                     Bfield=Bfield,
                     Region=Region,
                     Particles=Particles,
                     Medium=Medium,
                     InteractNUC=NuclearInteraction,
                     UseDecay=UseDecay,
                     Date=Date,
                     Step=1e-6,
                     Num=100000,
                     ForwardTrck=1,
                     BreakCondition=BreakConditions,
                     Save=Save,
                     Nfiles=Nfiles,
                     Output=Output,
                     Verbose=Verbose)
