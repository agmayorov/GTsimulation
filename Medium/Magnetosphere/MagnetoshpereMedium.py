from abc import ABC

import numpy as np
import datetime
import pyproj
import iri2016

from GeantInteraction import G4Interaction
from Global import vecRotMat, Constants
from Medium import GTGeneralMedium


class MagnetosphereMedium(GTGeneralMedium, ABC):

    def __init__(self):
        super().__init__()
        self.transformer = None
        self.GenMax = None
        self.IntPathDen = None

    def convert_xyz_to_lla(self, x, y, z):
        return self.transformer.transform(x, y, z, radians=False)

    def SetNuclear(self, InteractNuc):
        self.GenMax = InteractNuc["GenMax"]
        self.IntPathDen = InteractNuc.get("IntPathDen")

    def InteractNuc(self, LocalPathDen, LocalDen, V_normalized, PDG, M, T, nLocal, LocalChemComp):
        if self.IntPathDen is None:
            return

        if self.IntPathDen < LocalPathDen:
            rotationMatrix = vecRotMat(np.array([0, 0, 1]), V_normalized)
            rLocal, vLocal, Tint, status, process, product = G4Interaction(PDG, T / 1e3, LocalPathDen,
                                                                           LocalDen / nLocal,
                                                                           LocalChemComp / nLocal)

            if not status:
                Vm = rotationMatrix @ vLocal
                Tint *= 1e3
                assert 0.99 * Tint <= T
                T = Tint
                Vcorr = Constants.c * np.sqrt(1 - (M / (Tint + M)) ** 2) / np.linaglg.norm(Vm)
                Vm *= Vcorr

                return status, [Vm, np.linalg.norm(Vm), T]
