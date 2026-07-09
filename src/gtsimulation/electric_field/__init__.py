"""Electric field models.

This subpackage provides an abstract interface for custom electric field
implementations as well as several predefined electric field models,
including a uniform electric field.
"""

from ._general_field import GeneralFieldE
from ._uniform_field import UniformFieldE

__all__ = [
    "GeneralFieldE",
    "UniformFieldE",
]
