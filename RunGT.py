import numpy as np
from datetime import datetime

from MagneticFields import Regions
from GT.Algos import BunemanBorisSimulator

Region = Regions.Magnetosphere
Bfield = "Dipole"
# Bfield = ["Gauss", {'model': "IGRF", "model_type": "core", "version": 13}]
Date = datetime(2002, 12, 12)

Flux = ["Monolines", {"T": 1, "Center": np.array([0, 0, 0]), "Radius": 1, "Nevents": 10}]
# Flux = "Monolines"

simulator = BunemanBorisSimulator(Date=Date, Region=Region, Bfield=Bfield, Particles=Flux, Num=int(1e5), Step=1)
# print()
simulator()
