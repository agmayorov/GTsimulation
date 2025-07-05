from datetime import datetime

import numpy as np
from Scripts.pywin32_postinstall import verbose
from gtsimulation.Algos import BunemanBorisSimulator
from gtsimulation.Particle import Flux
from gtsimulation.Medium import GTUniformMedium
from gtsimulation.Particle.Generators import Monolines, SphereSurf
from gtsimulation.Global import Regions
from gtsimulation.Global import Units as U


date = datetime(2025, 1, 1)
bfield = None
efield = None
medium = GTUniformMedium()
use_decay = False
nuclear_interaction = None
rad_losses = False
region = Regions.Undefined
nevents = 5
particles = Flux(
    Spectrum = Monolines(energy = 1.5 * U.GeV),
    Distribution = SphereSurf(Radius = 0),
    Names = "proton",
    Nevents = nevents
)
output=None
nfiles = 1
berbose = True
break_conditions = None
save = 1 # saving all points

dt = 1e-2
steps = int(1e6)


simulator = BunemanBorisSimulator(Bfield=bfield,
                                  Efield=efield,
                                  Region=region,
                                  Particles=particles,
                                  Medium=medium,
                                  RadLosses=rad_losses,
                                  InteractNUC=nuclear_interaction,
                                  UseDecay=use_decay,
                                  Date=date,
                                  Step=dt,
                                  Num=steps,
                                  ForwardTrck=1,
                                  BreakCondition=break_conditions,
                                  Save=save,
                                  Nfiles=nfiles,
                                  Output=output,
                                  Verbose=verbose)

from timeit import default_timer as timer
st = timer()
simulator()
tot_time = timer() - st
print(f"Total time for {nevents} Events * {steps} Iterations = {tot_time} seconds")

