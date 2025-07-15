from datetime import datetime

import numpy as np
from gtsimulation.Algos import BunemanBorisSimulator
from gtsimulation.ElectricFields import UniformFieldE
from gtsimulation.Particle import Flux
from gtsimulation.Particle.Generators import Monolines, SphereSurf
from gtsimulation.Global import Regions
from gtsimulation.Global import Units as U


date = datetime(2025, 1, 1)
b_field = None
e_field = UniformFieldE(E=np.array([1000, 0, 0]))
medium = None
use_decay = False
nuclear_interaction = None
rad_losses = False
region = Regions.Undefined
n_events = 5
particles = Flux(
    Spectrum = Monolines(energy = 1.5 * U.GeV),
    Distribution = SphereSurf(Radius = 0),
    Names = "proton",
    Nevents = n_events
)
output = None
n_files = 1
verbose = True
break_conditions = None
save = 1 # saving all points

dt = 1e-2
steps = int(1e6)


simulator = BunemanBorisSimulator(
    Bfield=b_field,
    Efield=e_field,
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
    Nfiles=n_files,
    Output=output,
    Verbose=verbose
)

from timeit import default_timer as timer
st = timer()
simulator()
tot_time = timer() - st
print(f"Total time for {n_events} Events * {steps} Iterations = {tot_time} seconds")
