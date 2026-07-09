"""Particle interaction models.

This subpackage provides implementations of physical processes describing
particle interactions with matter, including nuclear interactions, radiation
losses, and photon emission.
"""

from .nuclear_interaction import NuclearInteraction
from .G4functions import *
from .GenSynchCounter import *
from .SynchrotronEmission import *

__all__ = [
    "NuclearInteraction",
]
