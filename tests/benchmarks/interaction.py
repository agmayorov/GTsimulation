from datetime import datetime
from timeit import default_timer as timer

import numpy as np
from gtsimulation.Algos import BunemanBorisSimulator
from gtsimulation.Particle import Flux
from gtsimulation.Medium import GTUniformMedium
from gtsimulation.MagneticFields import Uniform
from gtsimulation.Particle.Generators import Monolines, SphereSurf
from gtsimulation.Global import Regions
from gtsimulation.Global import Units as U

date = datetime(2025, 1, 1)
b_field = Uniform(B=np.array([0, 0, 10]))  # 10 nT
e_field = None

use_decay = False
nuclear_interaction = {"GenMax": 1}
region = Regions.Undefined
rad_losses = False
n_events = 1
particles = Flux(
    Spectrum=Monolines(energy=10*U.GeV),
    Distribution=SphereSurf(Radius=0),
    Names="mu+",
    Nevents=n_events
)
n_files = 1
verbose = False
break_conditions = None
save = [1, {"Energy": True, "Path": True}]  # saving all points

dt = 1e-2
steps = int(1e6)

rho_array = np.array([1e-10, 2e-10, 5e-10, 1e-9, 2e-9, 5e-9, 1e-8])
time_array = np.zeros_like(rho_array)

for i, rho in enumerate(rho_array):
    medium = GTUniformMedium(density=rho)
    output = f'interaction_{rho}'
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
    st = timer()
    simulator()
    tot_time = timer() - st
    time_array[i] = tot_time
    print(f"Total time for density {rho} kg/m3: {tot_time} seconds")

np.savez('interaction.npz', rho_array=rho_array, time_array=time_array)
