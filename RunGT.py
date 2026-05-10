import argparse
import numpy as np
from datetime import datetime

from gtsimulation.Global import Regions, Units as U
from gtsimulation.Algos import BunemanBorisSimulator
from gtsimulation.MagneticFields.Magnetosphere import Gauss
from gtsimulation.Particle import Flux, FluxPitchPhase, Generators
from gtsimulation.Medium import GTnrlmsis

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)

args = parser.parse_args()
seed = args.seed

np.random.seed(seed)
Date = datetime(2008, 1, 1)

Region = Regions.Magnetosphere
Bfield = Gauss(date=Date, model="CHAOS", model_type="core", version=7.18)

# Medium = None
Medium = GTnrlmsis(date=Date, version=0)

Particles = Flux(
    Spectrum = Generators.spectrum.Monolines(energy = 1.5 * U.GeV),
    Distribution = Generators.distribution.SphereSurf(Radius = 10 * U.RE),
    Names = "proton",
    Nevents = 10
)
# Particles = FluxPitchPhase(
#     Bfield = Bfield,
#     Pitch = np.pi / 4,
#     Phase = np.pi / 6,
#     Spectrum = Monolines(energy = 20 * U.GeV),
#     Distribution = SphereSurf(Radius = 0, Center = np.array([1.2, 0, 0]) * U.RE),
#     Names = "proton",
#     Nevents = 20,
#     Mode = "Outward"
# )

UseDecay = False
NuclearInteraction = None
# NuclearInteraction = {"GenMax": 3}

Nfiles = 2
Output = f"output/filename"
Save = [1, {"Clock": True, "Path": True}]

Verbose = 2
BreakConditions = None

simulator = BunemanBorisSimulator(
    Particles=Particles,
    Num=100000,
    Step=1e-6,
    Bfield=Bfield,
    Region=Region,
    Medium=Medium,
    InteractNUC=NuclearInteraction,
    UseDecay=UseDecay,
    Date=Date,
    ForwardTrck=1,
    BreakCondition=BreakConditions,
    Save=Save,
    Nfiles=Nfiles,
    Output=Output,
    Verbose=Verbose
)
simulator()
