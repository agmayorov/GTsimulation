"""Check initialization of GT's main classes without running simulations.

Run with --help for options and Geant4 setup.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import sys
import traceback

DATE = datetime(2020, 1, 1, 12)


def main():
    scope_note = (
        "Checks and recommendations are based only on initialization of GT's main classes.\n"
        "Simulation runs are not tested."
    )
    parser = argparse.ArgumentParser(
        description="Check GT initialization in the current environment (no calculations).",
        epilog="Before building GT with Geant4 and running checks, use the same terminal:\n"
               "  source /path/to/geant4/bin/geant4.sh\n\n" + scope_note,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="show help message")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="skip resource-intensive checks")
    parser.add_argument("--no-geant4", action="store_true",
                        help="disable Geant4 checks")
    parser.add_argument("--traceback", action="store_true", help="print full tracebacks on failure")
    args = parser.parse_args()
    passed, failed, skipped = [], [], []

    @contextmanager
    def check(name):
        try:
            yield
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            failed.append((name, reason))
            print(f"[FAIL] {name}", flush=True)
            print("       " + reason.replace("\n", "\n       "), flush=True)
            if args.traceback:
                traceback.print_exc(file=sys.stdout)
        else:
            passed.append(name)
            print(f"[ OK ] {name}", flush=True)

    print("GT compatibility check")
    print(f"Python {sys.version.split()[0]} — {sys.executable}")
    print("Initialization only; calculations are not tested.")
    print("[ OK ] passed    [FAIL] failed    [SKIP] not checked", flush=True)
    print("\n--- Environment ---", flush=True)

    # Imports stay inside blocks so import failures have named results too.
    with check("GT import"):
        import gtsimulation

    print("\n--- Particles ---", flush=True)
    for name in ("proton", "anti_proton", "e-", "mu-", "neutron", "deuteron", "alpha", "Fe-56"):
        with check(f"Particle: {name}"):
            from gtsimulation.particle import Particle
            Particle(Name=name)

    with check("Particle: PDG code"):
        from gtsimulation.particle import Particle
        Particle(PDG=2212)

    with check("CRParticle: kinetic energy"):
        from gtsimulation.particle import CRParticle
        CRParticle(Name="proton", r=[0., 0., 0.], v=[1., 0., 0.], T=100.)

    with check("CRParticle: total energy"):
        from gtsimulation.particle import CRParticle
        CRParticle(Name="proton", T=None, E=1100.)

    print("\n--- Distributions and energy spectra ---", flush=True)
    # Qualify names: both distribution and spectrum define UserInput.
    with check("SphereSurf"):
        from gtsimulation.particle.generator import distribution
        distribution.SphereSurf(Radius=10., Center=[0., 0., 0.])

    with check("SphereVol"):
        from gtsimulation.particle.generator import distribution
        distribution.SphereVol(Radius=10.)

    with check("Disk"):
        from gtsimulation.particle.generator import distribution
        distribution.Disk(Radius=10., Width=1.)

    with check("UserInput: coordinates and directions"):
        from gtsimulation.particle.generator import distribution
        distribution.UserInput(R0=[[0., 0., 0.]], V0=[[1., 0., 0.]])

    with check("ContinuumSpectrum: base container"):
        from gtsimulation.particle.generator import spectrum
        spectrum.ContinuumSpectrum(energy_min=100., energy_max=200.)

    with check("Monolines"):
        from gtsimulation.particle.generator import spectrum
        spectrum.Monolines(energy=100.)

    with check("PowerSpectrum"):
        from gtsimulation.particle.generator import spectrum
        spectrum.PowerSpectrum(energy_min=100., energy_max=200.,
                               energy_range_units="R", base="T", spectrum_index=-2.)

    with check("ForceField"):
        from gtsimulation.particle.generator import spectrum
        spectrum.ForceField(energy_min=100., energy_max=200., modulation_potential=500.)

    with check("Uniform: energy spectrum"):
        from gtsimulation.particle.generator import spectrum
        spectrum.Uniform(energy_min=100., energy_max=200.)

    with check("UserInput: energies"):
        from gtsimulation.particle.generator import spectrum
        spectrum.UserInput(energy=[100., 200.])

    print("\n--- Particle fluxes ---", flush=True)
    for mode in ("Inward", "Outward"):
        with check(f"Flux: {mode}"):
            from gtsimulation.particle import Flux, generator
            Flux(Spectrum=generator.Monolines(energy=100.),
                 Distribution=generator.SphereSurf(Radius=0.),
                 Names="proton", Nevents=1, Mode=mode)

    with check("Flux: mixed composition"):
        from gtsimulation.particle import Flux, generator
        Flux(Spectrum=generator.Monolines(energy=100.),
             Distribution=generator.SphereSurf(Radius=0.),
             Names=["proton", "e-"], Nevents=2)

    with check("FluxPitchPhase: angles in radians"):
        from gtsimulation.particle import FluxPitchPhase, generator
        from gtsimulation.magnetic_field import Uniform
        FluxPitchPhase(Bfield=Uniform(B=[0., 0., 100.]), Pitch=0.5, Phase=0.,
                       Spectrum=generator.Monolines(energy=100.),
                       Distribution=generator.SphereSurf(Radius=0.), Names="proton")

    with check("GyroCenterFlux: angles in degrees"):
        from gtsimulation.particle import GyroCenterFlux, generator
        from gtsimulation.magnetic_field import Uniform
        GyroCenterFlux(Bfield=Uniform(B=[0., 0., 100.]), Pitchd=45., Phased=0.,
                       CooGyr=[1e7, 0., 0.], Spectrum=generator.Monolines(energy=100.),
                       Distribution=generator.SphereSurf(Radius=0.), Names="proton")

    print("\n--- Uniform and summed fields ---", flush=True)
    # Fields. Optional components have separate results.
    with check("Uniform: magnetic field"):
        from gtsimulation.magnetic_field import Uniform
        Uniform(B=[0., 0., 100.])

    with check("Summed: magnetic fields"):
        from gtsimulation.magnetic_field import Summed, Uniform
        Summed(models=[Uniform(B=[100., 0., 0.]), Uniform(B=[0., 100., 0.])])

    with check("Uniform: electric field"):
        from gtsimulation.electric_field import Uniform
        Uniform(E=[1., 0., 0.])

    print("\n--- Magnetosphere fields ---", flush=True)
    with check("Dipole: constant"):
        from gtsimulation.magnetic_field.magnetosphere import Dipole
        Dipole()

    with check("Dipole: date-dependent coefficients"):
        from gtsimulation.magnetic_field.magnetosphere import Dipole
        Dipole(date=DATE)

    # Each coefficient file is checked independently, within its date interval.
    for body, model, kind, revision, year in (
        ("Earth", "IGRF", "core", 13, 2020),
        ("Earth", "IGRF", "core", 14, 2020),
        ("Earth", "CHAOS", "core", 7.18, 2020),
        ("Earth", "CHAOS", "static", 7.18, 2020),
        ("Earth", "CHAOS", "core", 8.5, 2020),
        ("Earth", "CHAOS", "static", 8.5, 2020),
        ("Earth", "CM", "core", 6, 2015),
        ("Earth", "CM", "static", 6, 2015),
        ("Earth", "CM", "ionosphere", 6, 2015),
        ("Earth", "COV_OBS", "core", 2, 2020),
        ("Earth", "LCS", "static", 1, 2020),
        ("Earth", "SIFM", "core", None, 2014),
        ("Earth", "SIFM", "static", None, 2014),
        ("Earth", "DIFI", "ionosphere", 6, 2020),
        ("Jupiter", "JRM09", "core", None, 2020),
        ("Jupiter", "JRM33", "core", None, 2020),
        ("Saturn", "Cassini11", "core", None, 2020),
        ("Saturn", "Cassini11plus", "core", None, 2020),
        ("Uranus", "Q3", "core", None, 2020),
        ("Neptune", "O8", "core", None, 2020),
        ("Mercury", "MBF_a_n", "core", None, 2020),
        ("Mars", "Langlais2019", "static", None, 2020),
    ):
        with check(f"Gauss: {body}/{model}/{kind}, version={revision}"):
            from gtsimulation.magnetic_field.magnetosphere import Gauss
            Gauss(date=datetime(year, 1, 1, 12), model=model, model_type=kind,
                  version=revision, planet=body)

    with check("Gauss: Ganymede/MagModel4"):
        from gtsimulation.magnetic_field.magnetosphere import Gauss
        Gauss(date=DATE, model="MagModel4", model_type="core", moon="Ganymede")

    for code in ("89", "96", "15B"):
        with check(f"Tsyganenko: T{code}"):
            from gtsimulation.magnetic_field.magnetosphere import Tsyganenko
            Tsyganenko(date=DATE, ModCode=code)

    print("\n--- Heliosphere fields ---", flush=True)
    with check("Parker: regular field without HCS"):
        from gtsimulation.magnetic_field.heliosphere import Parker
        Parker(date=DATE, use_noise=False, use_hcs=False)

    with check("Parker: regular field with HCS"):
        from gtsimulation.magnetic_field.heliosphere import Parker
        Parker(date=DATE, use_noise=False, use_hcs=True)

    for slab, two_d in ((True, False), (False, True), (True, True)):
        with check(f"Parker: noise, slab={slab}, 2D={two_d}"):
            from gtsimulation.magnetic_field.heliosphere import Parker
            Parker(date=DATE, use_noise=True, use_slab=slab, use_2d=two_d)

    with check("ParkerUniform: regular field"):
        from gtsimulation.magnetic_field.heliosphere import ParkerUniform
        ParkerUniform(x=1., y=0.1, z=0.2, use_noise=False)

    for slab, two_d in ((True, False), (False, True), (True, True)):
        with check(f"ParkerUniform: noise, slab={slab}, 2D={two_d}"):
            from gtsimulation.magnetic_field.heliosphere import ParkerUniform
            ParkerUniform(x=1., y=0.1, z=0.2, use_noise=True, use_slab=slab, use_2d=two_d)

    with check("UniformHelio: regular field"):
        from gtsimulation.magnetic_field.heliosphere import UniformHelio
        UniformHelio(use_noise=False)

    with check("UniformHelio: slab noise"):
        from gtsimulation.magnetic_field.heliosphere import UniformHelio
        UniformHelio(use_noise=True, use_slab=True, use_2d=False)

    if args.skip_heavy:
        skipped.extend(["UniformHelio: 2D noise (--skip-heavy)",
                        "UniformHelio: slab + 2D noise (--skip-heavy)"])
    else:
        with check("UniformHelio: 2D noise (large constructor allocations)"):
            from gtsimulation.magnetic_field.heliosphere import UniformHelio
            UniformHelio(use_noise=True, use_slab=False, use_2d=True)

        with check("UniformHelio: slab + 2D noise (large constructor allocations)"):
            from gtsimulation.magnetic_field.heliosphere import UniformHelio
            UniformHelio(use_noise=True, use_slab=True, use_2d=True)

    print("\n--- Galaxy fields ---", flush=True)
    with check("JF12mod: regular field"):
        from gtsimulation.magnetic_field.galaxy import JF12mod
        JF12mod(use_turb=False)

    if args.skip_heavy:
        skipped.append("JF12mod: turbulence (--skip-heavy)")
    else:
        with check("JF12mod: turbulence (large installed data file)"):
            from gtsimulation.magnetic_field.galaxy import JF12mod
            JF12mod(use_turb=True)

    for model in ("base", "expX", "neCL", "nebCor", "cre10", "synCG"):
        with check(f"UF23: {model}, regular field"):
            from gtsimulation.magnetic_field.galaxy import UF23
            UF23(model_type=model, use_turb=False)

    print("\n--- Medium and interaction ---", flush=True)
    with check("GTUniformMedium"):
        from gtsimulation.medium import GTUniformMedium
        GTUniformMedium(density=1e-10, element="H")

    with check("GTVacuum"):
        from gtsimulation.medium import GTVacuum
        GTVacuum()

    with check("InterstellarGas"):
        from gtsimulation.medium.galaxy import InterstellarGas
        InterstellarGas()

    for revision in (0, 2.0, 2.1):
        with check(f"GTnrlmsis: version={revision}, solar indices from pymsis"):
            from gtsimulation.medium.magnetosphere import GTnrlmsis
            GTnrlmsis(date=DATE, version=revision)

    with check("SynchCounter"):
        from gtsimulation.interaction import SynchCounter
        SynchCounter()

    print("\n--- Simulators and their options ---", flush=True)
    # Common input for the seven solvers. No particle generation or simulation.
    # If this prerequisite fails, dependent checks are explicitly skipped.
    particles = None
    with check("Flux: input for simulator constructors"):
        from gtsimulation.particle import Flux, generator
        particles = Flux(Spectrum=generator.Monolines(energy=100.),
                         Distribution=generator.SphereSurf(Radius=0.), Names="proton")

    if particles is None:
        skipped.append("Simulator constructors and options: input Flux failed")
    else:
        with check("BunemanBorisSimulator"):
            from gtsimulation.pusher import BunemanBorisSimulator
            BunemanBorisSimulator(Particles=particles, Num=1, Step=1e-4, Verbose=0)

        with check("VaySimulator"):
            from gtsimulation.pusher import VaySimulator
            VaySimulator(Particles=particles, Num=1, Step=1e-4, Verbose=0)

        with check("HigueraCarySimulator"):
            from gtsimulation.pusher import HigueraCarySimulator
            HigueraCarySimulator(Particles=particles, Num=1, Step=1e-4, Verbose=0)

        with check("RungeKutta4Simulator"):
            from gtsimulation.pusher import RungeKutta4Simulator
            RungeKutta4Simulator(Particles=particles, Num=1, Step=1e-4, Verbose=0)

        with check("RungeKutta6Simulator"):
            from gtsimulation.pusher import RungeKutta6Simulator
            RungeKutta6Simulator(Particles=particles, Num=1, Step=1e-4, Verbose=0)

        with check("RungeKutta4SimulatorFast"):
            from gtsimulation.pusher import RungeKutta4SimulatorFast
            RungeKutta4SimulatorFast(Particles=particles, Num=1, Step=1e-4, Verbose=0)

        with check("RungeKutta6SimulatorFast"):
            from gtsimulation.pusher import RungeKutta6SimulatorFast
            RungeKutta6SimulatorFast(Particles=particles, Num=1, Step=1e-4, Verbose=0)

        # These checks accept constructor options; they do not validate physics.
        with check("BunemanBorisSimulator: adaptive step"):
            from gtsimulation.pusher import BunemanBorisSimulator
            BunemanBorisSimulator(Particles=particles, Num=1, Verbose=0,
                                 Step={"UseAdaptiveStep": True, "InitialStep": 1e-4,
                                       "LarmorRad": 600, "MaxTimeStep": 1e-2})

        with check("BunemanBorisSimulator: radiation losses"):
            from gtsimulation.pusher import BunemanBorisSimulator
            BunemanBorisSimulator(Particles=particles, Num=1, Step=1e-4,
                                 RadLosses=True, Verbose=0)

        with check("BunemanBorisSimulator: photon settings"):
            from gtsimulation.pusher import BunemanBorisSimulator
            BunemanBorisSimulator(Particles=particles, Num=1, Step=1e-4, Verbose=0,
                                 RadLosses=[True, {"Photons": True, "MinE": 0., "MaxE": 100.}])

        with check("BunemanBorisSimulator: decay setting (backend not checked)"):
            from gtsimulation.pusher import BunemanBorisSimulator
            BunemanBorisSimulator(Particles=particles, Num=1, Step=1e-4,
                                 UseDecay=True, Verbose=0)

        with check("BunemanBorisSimulator: backtracking"):
            from gtsimulation.pusher import BunemanBorisSimulator
            BunemanBorisSimulator(Particles=particles, Num=1, Step=1e-4, Verbose=0,
                                 ForwardTrck=-1)

        with check("BunemanBorisSimulator: break conditions"):
            from gtsimulation.pusher import BunemanBorisSimulator
            BunemanBorisSimulator(Particles=particles, Num=1, Step=1e-4, Verbose=0,
                                 BreakCondition={"Rmax": 1e8})

        for name in ("Undefined", "Magnetosphere", "Heliosphere", "Galaxy"):
            with check(f"BunemanBorisSimulator: region={name}"):
                from gtsimulation.common import Regions
                from gtsimulation.pusher import BunemanBorisSimulator
                BunemanBorisSimulator(Particles=particles, Num=1, Step=1e-4,
                                     Region=Regions[name], Verbose=0)

    print("\n--- Geant4 ---", flush=True)
    # The constructor starts a worker. Sending its stop sentinel lets us check
    # worker initialization/exit without calculating any Geant4 events.
    if not args.no_geant4:
        with check("NuclearInteraction: constructor and Geant4 worker startup"):
            from gtsimulation.interaction import NuclearInteraction
            from gtsimulation.interaction._build_config import GEANT4_COMPONENTS_AVAILABLE
            if not GEANT4_COMPONENTS_AVAILABLE:
                raise RuntimeError(
                    "GT was built without Geant4 support. Source geant4.sh before rebuilding GT."
                )
            interaction = NuclearInteraction(seed=42)
            try:
                interaction.in_q.put(None)
                interaction.process.join(timeout=120)
                if interaction.process.is_alive():
                    raise TimeoutError("Geant4 worker did not initialize and exit within 120 s")
                if interaction.process.exitcode != 0:
                    raise RuntimeError(f"Geant4 worker exit code: {interaction.process.exitcode}")
            finally:
                if interaction.process.is_alive():
                    interaction.process.terminate()
                    interaction.process.join(timeout=5)
                interaction.in_q.close()
                interaction.out_q.close()
    else:
        skipped.append("NuclearInteraction (--no-geant4)")

    print("\n" + "=" * 64)
    print(f"RESULT: {len(passed)} passed | {len(failed)} failed | {len(skipped)} skipped")
    print("=" * 64)
    if failed:
        print("\nFailed checks:")
        for name, reason in failed:
            print(f"[FAIL] {name}")
            print("       " + reason.replace("\n", "\n       "))
    if skipped:
        print("\nSkipped checks:")
        for name in skipped:
            print(f"[SKIP] {name}")
    print()
    if failed:
        print("Recommendation: fix the failures before using the affected GT features.")
    else:
        print("Recommendation: selected GT features initialize successfully; validate calculations separately.")
    if not args.no_geant4 and any(name.startswith("NuclearInteraction:") for name, _ in failed):
        print("Geant4 setup (same terminal, before building GT and running checks):")
        print("  source /path/to/geant4/bin/geant4.sh")
        print("If GT was built without Geant4, rebuild it after sourcing this environment.")
    if skipped:
        print("Skipped features have not been validated.")
    print("\n" + scope_note)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())