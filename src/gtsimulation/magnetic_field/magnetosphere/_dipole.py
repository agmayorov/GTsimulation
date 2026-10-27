import datetime
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numba import njit

from gtsimulation.common import Units, Regions
from gtsimulation.magnetic_field import AbsBfield


class Dipole(AbsBfield):
    """
    Static magnetic dipole field.

    Parameters
    ----------
    B_eq : float or None, optional
        Equatorial field at the reference radius, in nT. If None, use the
        Earth normalization: 30100 nT, or the coefficient fit for date.
        Default is None.
    reference_radius : float, optional
        Radius at which B_eq is specified, in metres. Default is Units.RE.
    direction : array-like, shape (3,), optional
        Nonzero magnetic moment direction, normalized automatically.
        Default is (0, 0, -1), matching the Earth dipole polarity.
    center : array-like, shape (3,), optional
        Dipole center in metres. Default is (0, 0, 0).
    name : str or None, optional
        Object name for the GT logger. If None, use "Earth" when B_eq is
        omitted, otherwise "Custom dipole". Default is None.
    date : datetime.datetime or None, optional
        Date for the Earth normalization fit (1900--2021), used when B_eq
        is omitted. None selects the fixed 30100 nT value. Default is None.
    use_tesla : bool, optional
        If True, GetBfield returns tesla; otherwise it returns nT.
        Default is True.
    use_meters : bool, optional
        If True, GetBfield accepts coordinates in metres; otherwise it
        accepts coordinates in reference radii. Default is True.

    Notes
    -----
    CalcBfield accepts coordinates in units of reference_radius and returns
    magnetic field components in nT. GetBfield applies the coordinate and
    field conversions selected by use_meters and use_tesla.
    The reference radius sets the field normalization, not a surface boundary.
    """

    ToMeters = Units.RE2m

    def __init__(self,
                 B_eq: float | None = None,
                 reference_radius: float = Units.RE2m,
                 direction: Sequence[float] | np.ndarray = (0, 0, -1),
                 center: Sequence[float] | np.ndarray = (0, 0, 0),
                 name: str | None = None,
                 date: datetime.datetime | None = None, *,
                 use_tesla: bool = True,
                 use_meters: bool = True) -> None:
        super().__init__(use_tesla=use_tesla, use_meters=use_meters)
        self.Region = Regions.Magnetosphere
        self.ModelName = "Dipole"
        self.Units = "RE" if reference_radius == Units.RE2m else "R_ref"
        self.ToMeters = reference_radius
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.center = np.array(center, dtype=float)
        self.name = name or ("Earth" if B_eq is None else "Custom dipole")
        self.Date = date
        self.B_eq = 30100.0 if B_eq is None else B_eq
        self._earth_normalization = B_eq is None
        if date is not None:
            self.UpdateState(date)

    def from_meters(self, x, y, z):
        return x / self.ToMeters, y / self.ToMeters, z / self.ToMeters

    def CalcBfield(self, x, y, z, **kwargs):
        return self._calc_field(x - self.center[0] / self.ToMeters,
                                y - self.center[1] / self.ToMeters,
                                z - self.center[2] / self.ToMeters,
                                self.B_eq, self.direction)

    @staticmethod
    @njit(fastmath=True)
    def _calc_field(x, y, z, B_eq, direction):
        r2 = x * x + y * y + z * z
        scale = B_eq / (r2 * np.sqrt(r2))
        projection = (direction[0] * x + direction[1] * y + direction[2] * z) / r2
        return (scale * (3 * projection * x - direction[0]),
                scale * (3 * projection * y - direction[1]),
                scale * (3 * projection * z - direction[2]))

    def UpdateState(self, new_date: datetime.datetime):
        self.Date = new_date
        if self._earth_normalization:
            assert 1900 <= new_date.year <= 2021
            coefs = np.load(Path(__file__).parent / "Data" / "HarmonicCoeffsIGRF.npy",
                            allow_pickle=True).item()
            # Retain the original decimal-year convention for the Earth fit.
            days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if new_date.year % 4 == 0:
                days[1] = 29
            year = new_date.year + (new_date.day + sum(days[:new_date.month - 1]) - 0.5) / sum(days)
            self.B_eq = np.sqrt(sum(np.polyval(coefs[key], year) ** 2
                                   for key in ("g11_fit", "h11_fit", "g10_fit")))

    def __str__(self):
        return (
            "Dipole\n"
            f"\tObject: {self.name}\n"
            f"\tEquatorial field: {self.B_eq:g} nT\n"
            f"\tReference radius: {self.ToMeters:g} m\n"
            f"\tMoment direction: {self.direction}\n"
            f"\tCenter: {self.center} m"
        )
