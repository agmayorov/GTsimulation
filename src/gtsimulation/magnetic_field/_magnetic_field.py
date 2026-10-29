from abc import ABC, abstractmethod

from gtsimulation.common import Units, Regions


class AbsBfield(ABC):
    """
    Abstract base class for magnetic field models.

    A subclass implements the field model in calc_field in the native
    coordinate units of the model: for example, galactic models take
    coordinates in kpc. calc_field must return the magnetic field in nT.
    The class attribute to_meters is the native length unit in metres
    (e.g. ``Units.kpc``) and must be set in the subclass; the default value
    1 corresponds to metres. A subclass must also implement update_state.
    Fields can be combined with ``+``, which returns their vector sum.
    """

    to_meters = 1

    def __init__(self):
        self.Region = Regions.Undefined
        self.ModelName = None

    @abstractmethod
    def calc_field(self, x, y, z):
        """
        Calculate the magnetic field in the native units of the model.

        Parameters
        ----------
        x, y, z : float
            Coordinates in the native units of the model.

        Returns
        -------
        Bx, By, Bz : float
            Magnetic field components in nT.
        """
        pass

    @abstractmethod
    def update_state(self, new_date):
        """
        Update the model for a new date.

        Models that do not depend on the date implement this method as a
        no-op.

        Parameters
        ----------
        new_date : datetime.datetime
            New date.
        """
        pass

    def get_field(self, x, y, z):
        Bx, By, Bz = self.calc_field(x / self.to_meters, y / self.to_meters, z / self.to_meters)
        return Bx / Units.T2nT, By / Units.T2nT, Bz / Units.T2nT

    def __add__(self, other):
        if not isinstance(other, AbsBfield):
            return NotImplemented
        return _SummedBfield([self, other])


class _SummedBfield(AbsBfield):
    """Vector sum of magnetic fields, created by ``+``."""

    def __init__(self, models):
        super().__init__()
        self.ModelName = "Summed"
        self.models = []
        for model in models:
            if isinstance(model, _SummedBfield):
                self.models.extend(model.models)
            else:
                self.models.append(model)

    def calc_field(self, x, y, z):
        Bx_s, By_s, Bz_s = 0, 0, 0
        for model in self.models:
            Bx, By, Bz = model.calc_field(x / model.to_meters, y / model.to_meters, z / model.to_meters)
            Bx_s += Bx
            By_s += By
            Bz_s += Bz
        return Bx_s, By_s, Bz_s

    def update_state(self, new_date):
        for model in self.models:
            model.update_state(new_date)

    def __str__(self):
        s = self.ModelName
        for i, model in enumerate(self.models, 1):
            name, *params = str(model).split("\n")
            s += f"\n\tModel {i}: {name}"
            for param in params:
                s += f"\n\t{param}"
        return s
