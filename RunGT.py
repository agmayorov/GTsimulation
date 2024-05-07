import numpy as np
from datetime import datetime

from MagneticFields import Regions
from GT.Algos import BunemanBorisSimulator

Region = Regions.Magnetosphere
Bfield = "Dipole"
# Bfield = ["Gauss", {'model': "IGRF", "model_type": "core", "version": 13}]
Date = datetime(2002, 12, 12)

Flux = ["Monolines", {"T": 0.3, "Center": np.array([0, 0, 0]), "Radius": 10, "Nevents": 10}]
# Flux = ["PowerSpectrum", {"EnergyMin": 0.1, "EnergyMax": 0.5, "RangeUnits": 'T', "Base": 'R', "SpectrumIndex": -2.7, "Radius": 5, "Nevents": 5}]
# Flux = "PowerSpectrum"
# Flux = "Monolines"

simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Particles=Flux, Num=int(5e4), Step=0.1)
# print()
simulator()
