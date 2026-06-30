from ._magnetic_field import AbsBfield
from ._summed_field import Summed
from ._uniform import Uniform

from . import magnetosphere, Heliosphere, Galaxy

__all__ = [
    "AbsBfield",
    "Summed",
    "Uniform",
]
