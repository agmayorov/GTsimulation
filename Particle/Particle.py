from pathlib import Path

import numpy as np
from particle import Particle as ParticleDB

alias_list = [
    ({"nu_e"}, "nu(e)"),
    ({"nu_mu"}, "nu(mu)"),
    ({"nu_tau"}, "nu(tau)"),
    ({"neutron"}, "n"),
    ({"proton", "H1"}, "p"),
    ({"deuteron", "H2"}, "D2"),
    ({"triton", "H3"}, "T3"),
    ({"alpha"}, "He4")
]
alias_set = set().union(*[element[0] for element in alias_list])
alias_map = {}
for elements, replacement in alias_list:
    for element in elements:
        alias_map[element] = replacement

nudat = np.load(Path(__file__).resolve().parent.joinpath("nndc_nudat_data.npy"))

class Particle:
    def __init__(self, Name=None, PDG=None):
        if Name is not None:
            self.Name = Name
            name_parsed = self.__parse_name(Name)
            p = ParticleDB.from_name(name_parsed)
            self.PDG = p.pdgid.real
        elif PDG is not None:
            self.PDG = PDG
            p = ParticleDB.from_pdgid(PDG)
            if p.programmatic_name.endswith("_bar"):
                self.Name = "anti_" + p.programmatic_name.split("_bar")[0]
            elif p.programmatic_name.endswith(("_minus", "_plus", "_0")):
                self.Name = p.name
            else:
                self.Name = p.programmatic_name
        else:
            raise Exception("The input parameters must contain either a Name or a PDG code.")
        self.Z = p.charge
        self.M = p.mass if p.mass is not None else 0.0
        if p.pdgid.is_nucleus:
            self.A = p.pdgid.A
            self.tau = nudat[(nudat["z"] == np.abs(p.charge)) & (nudat["n"] == p.pdgid.A - np.abs(p.charge))]["halflife"].item() * np.sqrt(2)
        else:
            self.A = 0
            self.tau = p.lifetime * 1e-9 if p.lifetime is not None else np.inf

    @staticmethod
    def __parse_name(name):
        name_parsed = name
        if name.startswith("anti_"):
            _, name_parsed = name.split("anti_")
        elif name.endswith("_bar"):
            name_parsed, _ = name.split("_bar")
        if "-" in name_parsed and name_parsed.split("-")[1].isdigit():
            name_parsed = "".join(name_parsed.split("-"))
        if name_parsed in alias_set:
            name_parsed = alias_map.get(name_parsed)
        if name.startswith("anti_") or name.endswith("_bar"):
            name_parsed += "~"
        return name_parsed


class CRParticle(Particle):
    def __init__(self, r=np.array([1, 0, 0]), v=np.array([1, 0, 0]), T=1, E=None, **kwargs):
        super().__init__(**kwargs)
        self.coordinates = r
        self.velocities = v / np.linalg.norm(v)
        if T is not None and T > 0:
            self.T = T
            self.E = T + self.M
        elif E is not None and E > 0:
            self.E = E
            self.T = E - self.M
        else:
            raise Exception("Particle without or with negative energy")
