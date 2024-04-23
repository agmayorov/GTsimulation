import sys
from NucleiProp import NucleiProp


class Particle:

    def __init__(self, Type=None, Z=None, M=None, PDG=None, Name=None, E=None):
        try:
            # Kinematics
            if E is None or E <= 0:
                raise ValueError
            if isinstance(E, float) or isinstance(E, int):
                self.E = E  # Mono-lines

            # TODO: add others functions of distribution of energy
            #   elif isinstance(E, str):

            # Self properties
            if Type is not None and Z is None and M is None and PDG is None and Name is None:
                for key in NucleiProp:
                    if key == Type:
                        self.A = NucleiProp[key]['A']
                        self.Z = NucleiProp[key]['Z']
                        self.M = NucleiProp[key]['M']
                        self.PDG = NucleiProp[key]['PDG']
                        self.Name = key
                        break
            elif Type is None and Z is None and M is None and PDG is not None and Name is None:
                for key in NucleiProp:
                    if NucleiProp[key]['PDG'] == PDG:
                        self.A = NucleiProp[key]['A']
                        self.Z = NucleiProp[key]['Z']
                        self.M = NucleiProp[key]['M']
                        self.PDG = NucleiProp[key]['PDG']
                        self.Name = key
                        break
            elif Type is None and Z is not None and M is not None and PDG is None and Name is not None:
                self.A = None
                self.Z = Z
                self.M = M
                self.PDG = None
                self.Name = Name
            else:
                raise SyntaxError

        except SyntaxError:
            print('Invalid syntax in constructor')
        except ValueError:
            print('Invalid value of an argument')
