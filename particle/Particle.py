from particle.NucleiProp import NucleiProp


class Particle:

    def __init__(self, Type=None, Z=None, M=None, PDG=None, Name=None, E=None):
        try:
            # Kinematics
            if E is None or E <= 0:
                raise ValueError
            if isinstance(E, float) or isinstance(E, int):
                self.__E = E  # Mono-lines

            # TODO: add others functions of distribution of energy
            #   elif isinstance(E, str):

            # Self properties
            if Type is not None and Z is None and M is None and PDG is None and Name is None:
                for key in NucleiProp:
                    if key == Type:
                        self.__A = NucleiProp[key]['A']
                        self.__Z = NucleiProp[key]['Z']
                        self.__M = NucleiProp[key]['M']
                        self.__PDG = NucleiProp[key]['PDG']
                        self.__Name = key
                        break
            elif Type is None and Z is None and M is None and PDG is not None and Name is None:
                for key in NucleiProp:
                    if NucleiProp[key]['PDG'] == PDG:
                        self.__A = NucleiProp[key]['A']
                        self.__Z = NucleiProp[key]['Z']
                        self.__M = NucleiProp[key]['M']
                        self.__PDG = NucleiProp[key]['PDG']
                        self.__Name = key
                        break
            elif Type is None and Z is not None and M is not None and PDG is None and Name is not None:
                self.__A = None
                self.__Z = Z
                self.__M = M
                self.__PDG = None
                self.__Name = Name
            else:
                raise SyntaxError

        except SyntaxError:
            print('Invalid syntax in constructor')
        except ValueError:
            print('Invalid value of an argument')

    @property
    def A(self):
        return self.__A

    @property
    def Z(self):
        return self.__Z

    @property
    def M(self):
        return self.__M

    @property
    def PDG(self):
        return self.__PDG

    @property
    def Name(self):
        return self.__Name

    @property
    def E(self):
        return self.__E

class CRParticle(Particle):

    def __init__(self, r, v, Type=None, Z=None, M=None, PDG=None, Name=None, E=None):
        self.__r = r
        self.__v = v

        Particle.__init__(self, Type=Type, Z=Z, M=M, PDG=PDG, Name=Name, E=E)

    @property
    def coordinates(self):
        return self.__r

    @property
    def velocities(self):
        return self.__v