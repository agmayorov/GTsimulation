"""Particle transport simulation framework.

This package provides tools for simulating charged particle propagation in
physical environments, including trajectory integration, electromagnetic
fields, particle interactions, and medium descriptions.
"""

from ._simulator import GTSimulator
from . import pusher, electric_field, common, interaction, magnetic_field, medium, particle

__all__ = ["GTSimulator"]
