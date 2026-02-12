from datetime import datetime
from timeit import default_timer as timer

import numpy as np
from gtsimulation.Algos import BunemanBorisSimulator
from gtsimulation.Global import Regions, Units as U
from gtsimulation.MagneticFields import Uniform
from gtsimulation.Medium import GTUniformMedium
from gtsimulation.Interaction import NuclearInteraction
from gtsimulation.Particle import Flux, Generators

def get_number_of_secondaries(gt_output):
    n = np.zeros(len(gt_output))
    for k, primary in enumerate(gt_output):
        secondaries = primary['Child']
        n[k] = len(secondaries)
        if secondaries:
            n[k] += np.sum(get_number_of_secondaries(secondaries))
    return n

def get_number_of_generations(gt_output):
    n = np.ones(len(gt_output))
    for k, primary in enumerate(gt_output):
        secondaries = primary['Child']
        if secondaries:
            n[k] += np.max(get_number_of_generations(secondaries))
    return n


date = datetime(2025, 1, 1)
b_field = Uniform(B=np.array([0, 0, 10])) # 10 nT
e_field = None
medium = GTUniformMedium(density=1e-9)
use_decay = True
nuclear_interaction = NuclearInteraction(max_generations=20, seed=42)
region = Regions.Undefined
rad_losses = False
n_events = 10
output = None
verbose = 0
break_conditions = None
save = 0 # without saving points

dt = 1e-2
steps = int(1e6)

energy = np.array([1, 2, 5, 10, 20, 50, 100], dtype=np.float64)
time = np.zeros_like(energy)
n_secondaries = np.zeros_like(energy)
n_generations = np.zeros_like(energy)

for i, e in enumerate(energy):
    particles = Flux(
        Spectrum=Generators.Monolines(energy=e * U.GeV),
        Distribution=Generators.SphereSurf(Radius=0),
        Names="proton",
        Nevents=n_events
    )
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
        Output=output,
        Verbose=verbose
    )
    st = timer()
    sim_out = simulator()[0]
    time[i] = (timer() - st) / n_events
    n_secondaries[i] = np.mean(get_number_of_secondaries(sim_out))
    n_generations[i] = np.mean(get_number_of_generations(sim_out))
    print(f"Energy {e} GeV : total time {time[i]} seconds : "
          f"{n_secondaries[i]} secondaries : {n_generations[i]} generations")

np.savez('secondary.npz', energy=energy, time=time, n_secondaries=n_secondaries, n_generations=n_generations)
