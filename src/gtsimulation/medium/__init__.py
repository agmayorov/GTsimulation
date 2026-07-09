"""Medium models.

This subpackage provides an abstract interface for custom medium
implementations as well as several predefined models describing physical
media used in simulations.
"""

from ._general_medium import GTGeneralMedium
from ._uniform_medium import GTUniformMedium, GTVacuum

from . import galaxy, magnetosphere

__all__ = [
    "GTGeneralMedium",
    "GTUniformMedium",
    "GTVacuum",
]
