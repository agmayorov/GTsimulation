"""Electric field models.

This subpackage provides an abstract interface for custom electric field
implementations as well as several predefined electric field models,
including a uniform electric field.
"""

from ._general_field import GeneralFieldE
from ._uniform import Uniform

__all__ = [
    "GeneralFieldE",
    "Uniform",
]
