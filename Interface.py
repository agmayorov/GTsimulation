import importlib
import numpy as np


def InterfaceGT(Simulator=None,
                Bfield=None,
                Efield=None,
                Region=None,
                Medium=None,
                Date=None,
                RadLoss=False,
                Particles=None,
                TrackParams=False,
                ParticleOrigin=False,
                IsFirstRun=True,
                ForwardTrck=None,
                Save=1,
                Num=1e6,
                Step=1,
                Nfiles=1,
                Output=None,
                Verbose=False,
                BreakCondition=None,
                UseDecay=False,
                InteractNUC=None):
    #  Define Region
    Region = __SetRegion(Region)

    #  Define ElectroMagnetic Field
    Bfield, Efield = __SetEMFF(Bfield, Efield, Date, Region)

    #  Define Medium
    Medium = __SetMedium(Medium, Date, Region)

    #  Define Particles' Initial Conditions
    Particles = __SetParticles(Particles, Bfield, Region)

    # Define Simulator
    Simulator = __SetSimulator(Simulator)

    # Call GT
    return Simulator(Bfield, Efield, Region, Medium, Date, RadLoss, Particles, TrackParams, ParticleOrigin, IsFirstRun,
                     ForwardTrck, Save, Num, Step, Nfiles, Output, Verbose, BreakCondition, UseDecay, InteractNUC)()


def __SetRegion(Region):
    global_module = importlib.import_module("Global")
    if not isinstance(Region, list):
        Region = getattr(global_module, "Regions")[Region]
    else:
        name = Region[0]
        params = Region[1]
        Region = getattr(global_module, "Regions")[name]
        Region.value.set_params(**params)

    return Region


def __SetEMFF(Bfield, Efield, Date, Region):
    if Bfield is not None:
        module_name = f"MagneticFields.{Region.name}"
        m = importlib.import_module(module_name)
        class_name = Bfield if not isinstance(Bfield, list) else Bfield[0]
        params = {"date": Date, **({} if not isinstance(Bfield, list) else Bfield[1])}
        if hasattr(m, class_name):
            B = getattr(m, class_name)
            Bfield = B(**params)
        else:
            raise Exception("No such field")

    return Bfield, Efield


def __SetMedium(Medium, Date, Region):
    if Medium is not None:
        module_name = f"Medium.{Region.name}"
        m = importlib.import_module(module_name)
        class_name = Medium if not isinstance(Medium, list) else Medium[0]
        params = {"date": Date, **({} if not isinstance(Medium, list) else Medium[1])}
        if hasattr(m, class_name):
            class_medium = getattr(m, class_name)
            Medium = class_medium(**params)
        else:
            raise Exception("No such medium")

    return Medium


def __SetParticles(Particles, Bfield, Region):
    module_name = f"Particle.Generators"
    m = importlib.import_module(module_name)
    useB = False
    if isinstance(Particles, list):
        class_name = Particles[0]
        Particles = Particles[1]
        flux_class = getattr(importlib.import_module("Particle"), class_name)
        useB = True
    else:
        flux_class = getattr(importlib.import_module("Particle"), 'Flux')
    spectrum = Particles.pop("Spectrum", None)
    if spectrum is not None:
        if hasattr(m, spectrum):
            spectrum = getattr(m, spectrum)
            Particles["Spectrum"] = spectrum
        else:
            raise Exception("No spectrum")

    distribution = Particles.pop("Distribution", None)
    if distribution is not None:
        if hasattr(m, distribution):
            distribution = getattr(m, distribution)
            Particles["Distribution"] = distribution
        else:
            raise Exception("No Distribution")
    transform = Particles.pop("Transform", None)
    if transform is not None:
        center = Particles.get("Center", None)
        assert center is None
        center = Region.value.transform(*transform[1], transform[0])
        Particles["Center"] = np.array(center)
    params = {**Particles}
    if useB:
        params["Bfield"] = Bfield
    return flux_class(**params)


def __SetSimulator(Simulator):
    simulator_module = importlib.import_module("GT.Algos")
    return getattr(simulator_module, Simulator if Simulator.endswith("Simulator") else Simulator + "Simulator")


if __name__ == "__main__":
    from Global import Regions
    from datetime import datetime
    from MagneticFields.Magnetosphere import Gauss

    Date = datetime(2008, 1, 1)
    Region = Regions.Magnetosphere
    Bfield = Gauss(date=Date, model='CHAOS', model_type='core', version=7.13)

    print(__SetParticles({
        "Pitch": np.pi/4,
        "Nevents": 20,
        "Names": 'pr'
    }, Bfield, Region))
