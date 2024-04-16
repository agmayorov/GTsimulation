import os
from enum import Enum

from MagneticFields import AbsBfield, Regions


class GaussModels(Enum):
    IGRF = 1
    CHAOS = 2
    CM = 3
    COV_OBS = 4
    LCS = 5
    SIFM = 6
    DIFI = 7


class GaussTypes(Enum):
    core = 1
    static = 2
    ionosphere = 3


versions_dict = {GaussModels.IGRF: [13],
                 GaussModels.CHAOS: [7.13],
                 GaussModels.CM: [6],
                 GaussModels.COV_OBS: [2],
                 GaussModels.LCS: [1],
                 GaussModels.DIFI: [6],
                 GaussModels.SIFM: [None]}


class Gauss(AbsBfield):

    def __init__(self, model: GaussModels, tp: GaussTypes, ver=None):
        super().__init__()
        self.Region = Regions.Magnetosphere
        self.Model = model
        self.type = tp
        self.version = ver
        self.txt_file_loc = None
        self.mat_file_loc = None
        self.npy_file_loc = None
        self.SetFullModelName()

    def GetBfield(self, x, y, z, **kwargs):
        pass

    def UpdateState(self, new_date):
        pass

    def SetFullModelName(self):
        assert self.version in versions_dict[self.Model]
        txt_file = ""
        mat_file = ""
        npy_file = ""
        loc = os.path.dirname(os.path.realpath(__file__))
        if self.Model == GaussModels.IGRF:
            self.ModelName = self.Model.name + str(self.version)
            assert self.type == GaussTypes.core
            txt_file = self.ModelName.lower() + 'coeffs.txt'
            mat_file = self.ModelName.lower() + 'coeffs.mat'
            npy_file = self.ModelName.lower() + 'coeffs.npy'
        elif self.Model == GaussModels.CHAOS:
            assert self.type != GaussTypes.ionosphere
            txt_file = self.ModelName + "_" + self.type.name + '.shc.txt'
            mat_file = self.ModelName + "_" + self.type.name + '.mat'
            npy_file = self.ModelName + "_" + self.type.name + '.npy'
        elif self.Model == GaussModels.CM:
            self.ModelName = self.Model.name + str(self.version)
            if self.type == GaussTypes.core:
                txt_file = 'MCO_' + self.ModelName + '.shc.txt'
                mat_file = 'MCO_' + self.ModelName + '.mat'
                npy_file = 'MCO_' + self.ModelName + '.npy'
            elif self.type == GaussTypes.static:
                txt_file = 'MLI_' + self.ModelName + '.shc.txt'
                mat_file = 'MLI_' + self.ModelName + '.mat'
                npy_file = 'MLI_' + self.ModelName + '.npy'
            elif self.type == GaussTypes.ionosphere:
                txt_file = 'MIO_' + self.ModelName + '.shc.txt'
                mat_file = 'MIO_' + self.ModelName + '.mat'
                npy_file = 'MIO_' + self.ModelName + '.npy'
        elif self.Model == GaussModels.COV_OBS:
            self.ModelName = self.Model.name + ".x" + str(self.version) + "-int"
            txt_file = self.ModelName + '.shc.txt'
            mat_file = self.ModelName + '.mat'
            npy_file = self.ModelName + '.npy'
        elif self.Model == GaussModels.LCS:
            self.ModelName = self.Model.name + '-' + str(self.version)
            assert self.type == GaussTypes.static
            txt_file = self.ModelName + '.shc.txt'
            mat_file = self.ModelName + '.mat'
            npy_file = self.ModelName + '.npy'
        elif self.Model == GaussModels.DIFI:
            self.ModelName = self.Model.name + str(self.version)
            assert self.type == GaussTypes.ionosphere
            txt_file = self.ModelName + '.txt'
            mat_file = self.ModelName + '.mat'
            npy_file = self.ModelName + '.npy'
        elif self.Model == GaussModels.SIFM:
            self.ModelName = self.Model.name
            assert self.type != GaussTypes.ionosphere
            txt_file = self.ModelName + ".shc.txt"
            mat_file = self.ModelName + "_" + self.type.name + ".mat"
            npy_file = self.ModelName + "_" + self.type.name + ".npy"

        self.txt_file_loc = loc + os.sep + self.ModelName + os.sep + txt_file
        self.mat_file_loc = loc + os.sep + self.ModelName + os.sep + mat_file
        self.npy_file_loc = loc + os.sep + self.ModelName + os.sep + npy_file


