from datetime import datetime
from pathlib import Path
import numpy as np

from Global import Units, Regions
from MagneticFields import AbsBfield
from MagneticFields.Magnetosphere.Functions import transformations, t89, t96, gauss


class Tsyganenko(AbsBfield):
    """
    Tsyganenko empirical magnetospheric magnetic field model (T89/T96).

    This class provides an interface to compute the magnetic field components
    in the Earth's magnetosphere using the Tsyganenko empirical models (T89 or T96).
    The model type is selected via the `ModCode` parameter.

    :param date: Date for model initialization (default: 2000-01-01)
    :type date: datetime.datetime
    :param ModCode: Model version selector (89 or 96)
    :type ModCode: int
    :param kwargs: Additional arguments for base class
    :raises ValueError: If date is outside model's valid range
    """
    ToMeters = Units.RE2m

    def __init__(self, date: datetime = datetime(2000, 1, 1), ModCode=96, **kwargs):
        super().__init__(**kwargs)
        self.ModelName = "Tsyganenko"
        self.Region = Regions.Magnetosphere
        self.Units = "RE"
        self.mod_code = ModCode
        self.T_input = np.load(Path(__file__).resolve().parent.joinpath("Data", "T_input_short.npy"))
        if not self.T_input['date'][0] <= date <= self.T_input['date'][-1]:
            raise ValueError(f"Input date must be between {self.T_input['date'][0]} and {self.T_input['date'][-1]}.")
        self.Date = date
        self.__set_psi_ind()

    def __set_psi_ind(self):
        self.year = self.Date.year
        self.doy = self.Date.timetuple().tm_yday
        self.sec = self.Date.second + 60 * self.Date.minute + 3600 * self.Date.hour
        self.DTnum = self.Date.toordinal()  # + 366
        self.ps = self._get_psi()
        self.iopt = self._get_ind()

    def _get_psi(self):
        self.g, self.h, _ = gauss.LoadGaussCoeffs(Path(__file__).resolve().parent.joinpath("IGRF13", "igrf13coeffs.npy"), self.Date)
        [x, y, z] = transformations.geo2mag_eccentric(0, 0, 1, 0, self.g, self.h)
        [x, y, z] = transformations.gei2geo(x, y, z, self.year, self.doy, self.sec, 0)
        [x, y, z] = transformations.gei2gsm(x, y, z, self.year, self.doy, self.sec, 1)
        psi = np.arccos(z / np.linalg.norm([x, y, z]))
        return psi[0]

    def _get_ind(self):
        ia = np.argmax(self.T_input['date'] >= self.Date)
        match self.mod_code:
            case 89:
                return self.T_input['T89_input'][ia]
            case 96:
                return self.T_input['T96_input'][ia].astype(np.float32)
            case _:
                return None

    def CalcBfield(self, x, y, z, **kwargs):
        X, Y, Z = transformations.geo2gsm(x, y, z, self.year, self.doy, self.sec, 1)
        Bx, By, Bz = 0, 0, 0
        if self.mod_code == 89:
            Bx, By, Bz = t89.t89(self.iopt, self.ps, X, Y, Z)
        elif self.mod_code == 96:
            Bx, By, Bz = t96.t96(self.iopt, self.ps, X, Y, Z)
        Bx, By, Bz = transformations.geo2gsm(Bx, By, Bz, self.year, self.doy, self.sec, 0)
        return Bx, By, Bz

    def UpdateState(self, new_date):
        self.Date = new_date
        self.__set_psi_ind()

    def to_string(self):
        s = f"""Tsyganenko
        Mode: {self.mod_code}
        Psi: {self.ps}"""
        return s
