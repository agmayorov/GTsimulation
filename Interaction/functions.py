import subprocess
import warnings
import re
import numpy as np


def G4Interaction(PDG, E, m, rho, w):
    if np.sum(w) < 0.999:
        raise Exception("G4Int: total sum of medium fractions is not equal 1")

    if len(w) != 5:  # H He N O Ar
        raise Exception("G4Int: wrong number of fractions (atmosphere)")

    res = subprocess.run(f". /lustre/incos/set_pam_env.sh 10 4.10.06.p03; "
                         f"/lustre/mFunctions/G4Interaction/AtmosphericLayer/build/AtmosphericLayer "
                         f"{PDG} {E} {m} {rho} {str(w)[1:-1].replace(',', '')}", shell=True, stdout=subprocess.PIPE)
    if res.returncode != 0:
        print(res.stdout)
        raise Exception("Geant4 program did not work successfully")

    output = res.stdout
    output = output.decode("utf-8")

    k = output.find('Information about the primary particle:')
    if k == -1:
        raise RuntimeError('Primary particle information not found')

    segment = output[k:]

    position_match = re.search(r'Position: \((.+?),(.+?),(.+?)\)', segment)
    momentum_match = re.search(r'Momentum direction: \((.+?),(.+?),(.+?)\)', segment)
    kinetic_energy_match = re.search(r'Kinetic energy: (.+)', segment)
    step_status_match = re.search(r'Step status: (.+)', segment)
    process_name_match = re.search(r'Process name: (.+)', segment)

    if not (position_match and momentum_match and kinetic_energy_match and step_status_match and process_name_match):
        raise RuntimeError('Primary particle information format incorrect')

    r = np.array([float(position_match.group(1)), float(position_match.group(2)),
                  float(position_match.group(3))]) / 1e3  # [mm] to [m]
    v = np.array([float(momentum_match.group(1)), float(momentum_match.group(2)), float(momentum_match.group(3))])
    E_end = float(kinetic_energy_match.group(1)) / 1e3  # [MeV] to [GeV]
    status = int(step_status_match.group(1))
    process = process_name_match.group(1).strip()

    # Reading information about the secondary particles
    product = []

    k = [m.start() for m in re.finditer('Information about the secondary particle:', output)]
    if k:
        for start in k:
            segment = output[start:]
            particle_name_match = re.search(r'Particle name: (.+)', segment)
            momentum_match = re.search(r'Momentum direction: \((.+?),(.+?),(.+?)\)', segment)
            kinetic_energy_match = re.search(r'Kinetic energy: (.+)', segment)
            pdg_match = re.search(r'PDG: (.+)', segment)
            mass_match = re.search(r'Mass: (.+)', segment)
            charge_match = re.search(r'Charge: (.+)', segment)

            if particle_name_match and momentum_match and kinetic_energy_match and pdg_match and mass_match and charge_match:
                particle_name = particle_name_match.group(1)
                momentum_direction = [float(momentum_match.group(1)), float(momentum_match.group(2)),
                                      float(momentum_match.group(3))]
                kinetic_energy = float(kinetic_energy_match.group(1)) / 1e3
                pdg = float(pdg_match.group(1))
                mass = float(mass_match.group(1))
                charge = float(charge_match.group(1))

                product.append({
                    'ParticleName': particle_name,
                    'v': momentum_direction,
                    'E': kinetic_energy,
                    'PDG': [pdg, mass, charge]
                })

    return r, v, E_end, status, process, product


def G4Decay(PDG, E):
    res = subprocess.run(f'. /lustre/incos/set_pam_env.sh 10 4.10.06.p03; '
                         f'/lustre/mFunctions/G4Interaction/DecayGenerator/build/DecayGenerator {PDG} {E}',
                         shell=True, stdout=subprocess.PIPE)
    if res.returncode != 0:
        print(res.stdout)
        raise Exception("Geant4 program did not work successfully")

    output = res.stdout
    output = output.decode("utf-8")

    product = []

    # Find all occurrences of the secondary particle information
    k = [m.start() for m in re.finditer('Information about the secondary particle:', output)]

    if not k:
        warnings.warn('No decay products were found for the particle you entered')
        product.append({
            'ParticleName': PDG,
            'v': [0, 0, 1],
            'E': E,
            'PDG': []
        })
    else:
        for start in k:
            segment = output[start:]
            particle_name = re.search(r'Particle name: (.+)', segment).group(1)
            momentum = re.search(r'Momentum direction: \((.+?),(.+?),(.+?)\)', segment)
            momentum_direction = [float(momentum.group(1)), float(momentum.group(2)), float(momentum.group(3))]
            kinetic_energy = float(re.search(r'Kinetic energy: (.+)', segment).group(1)) / 1e3
            pdg = float(re.search(r'PDG: (.+)', segment).group(1))
            mass = float(re.search(r'Mass: (.+)', segment).group(1))
            lifetime = float(re.search(r'Particle LifeTime: (.+)', segment).group(1))
            charge = float(re.search(r'Charge: (.+)', segment).group(1))

            lifetime_scaled = lifetime * 1e-9 if lifetime > 0 else 0

            product.append({
                'ParticleName': particle_name,
                'v': momentum_direction,
                'E': kinetic_energy,
                'PDG': [pdg, mass, lifetime_scaled, charge]
            })

    return product


def G4Shower(PDG, E, h, alpha, doy, sec, lat, lon, f107A, f107, ap):
    res = subprocess.run(f". /lustre/incos/set_pam_env.sh 11 4.11.00.p02; "
                         f"/lustre/mFunctions/G4Interaction/AtmosphericColumn/build/AtmosphericColumn {PDG} {E} {h} "
                         f"{alpha} {doy} {sec} {lat} {lon} {f107A} {f107} {ap}", shell=True, stdout=subprocess.PIPE)
    if res.returncode != 0:
        print(res.stdout)
        raise Exception("Geant4 program did not work successfully")

    output = res.stdout
    output = output.decode("utf-8")

    k = output.find('Information about the primary particle:')
    if k == -1:
        raise RuntimeError('Primary particle information not found')

    segment = output[k:]

    name_match = re.search(r'Particle name: (.+)', segment)
    position_match = re.search(r'Position of interaction: \((.+?),(.+?),(.+?)\)', segment)
    process_name_match = re.search(r'Process name: (.+)', segment)

    if not (position_match and name_match and process_name_match):
        raise RuntimeError('Primary particle information format incorrect')

    r_int = np.array([float(position_match.group(1)), float(position_match.group(2)),
                      float(position_match.group(3))]) / 1e6  # [mm] to [km]
    ParticleName = name_match.group(1).strip()
    process = process_name_match.group(1).strip()

    albedo = []

    k = [m.start() for m in re.finditer('Albedo particles:', output)]
    if k:
        segment_albedo = output[k[0] + 18:]
        for seg in segment_albedo.split("\n")[:-1]:
            params = seg.split()
            particle_name = params[0]
            radius = [float(n) / 1e6 for n in params[1][1:-1].split(",")]
            momentum_direction = [float(n) for n in params[2][1:-1].split(",")]
            kinetic_energy = float(params[3]) / 1e3
            pdg, mass, charge = float(params[4]), float(params[5]), float(params[6])

            albedo.append({
                'ParticleName': particle_name,
                'r': radius,
                'v': momentum_direction,
                'E': kinetic_energy,
                'PDG': [pdg, mass, charge]
            })

    return ParticleName, r_int, process, albedo
