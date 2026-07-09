"""Particle definitions and initial conditions.

This subpackage provides tools for defining particle properties and generating
initial particle states, including spatial, energy, and velocity distributions.
"""

from ._particle import Particle, CRParticle
from ._flux import Flux, FluxPitchPhase, GyroCenterFlux
from .functions import ConvertT2R, ConvertR2T, GetAntiParticle

from . import generator

__all__ = [
    "Particle",
    "CRParticle",
    "Flux",
    "FluxPitchPhase",
    "GyroCenterFlux",
]
