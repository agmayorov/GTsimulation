"""Common utilities and definitions used throughout the gtsimulation package.

This subpackage provides a collection of shared components that are not tied
to a specific simulation module. It includes physical constants, unit
definitions, coordinate origins, utility functions, and common enumerations
used for controlling simulation processes and data handling.
"""

from .codes import BreakCode, BreakIndex, BreakDef, BreakMetric, SaveCode, SaveDef, SaveMetric
from .functions import vecRotMat
from .regions import Regions

from ._consts import Constants, Units, Origins

__all__ = [
    "Constants",
    "Units",
    "Origins",
]
