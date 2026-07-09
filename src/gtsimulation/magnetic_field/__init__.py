"""Magnetic field models.

This subpackage provides an abstract interface for custom magnetic field
implementations as well as several predefined magnetic field models,
including a uniform magnetic field.
"""

from ._magnetic_field import AbsBfield
from ._summed_field import Summed
from ._uniform import Uniform

from . import magnetosphere, heliosphere, galaxy

__all__ = [
    "AbsBfield",
    "Summed",
    "Uniform",
]
