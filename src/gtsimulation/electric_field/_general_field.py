from abc import ABC, abstractmethod

from gtsimulation.common import Regions


class GeneralFieldE(ABC):
    """
    Abstract base class for electric field models.

    A subclass implements the field model in calc_field in the native
    coordinate units of the model. calc_field must return the electric
    field in V/m. The class attribute to_meters is the native length unit
    in metres (e.g. ``Units.km``) and must be set in the subclass; the
    default value 1 corresponds to metres. Fields can be combined with
    ``+``, which returns their vector sum.
    """

    to_meters = 1

    def __init__(self):
        self.region = Regions.Undefined
        self.model_name = None

    @abstractmethod
    def calc_field(self, x, y, z):
        """
        Calculate the electric field in the native units of the model.

        Parameters
        ----------
        x, y, z : float
            Coordinates in the native units of the model.

        Returns
        -------
        Ex, Ey, Ez : float
            Electric field components in V/m.
        """
        pass

    def get_field(self, x, y, z):
        Ex, Ey, Ez = self.calc_field(x / self.to_meters, y / self.to_meters, z / self.to_meters)
        return Ex, Ey, Ez

    def __add__(self, other):
        if not isinstance(other, GeneralFieldE):
            return NotImplemented
        return _SummedEfield([self, other])


class _SummedEfield(GeneralFieldE):
    """Vector sum of electric fields, created by ``+``."""

    def __init__(self, models):
        super().__init__()
        self.model_name = "Summed"
        self.models = []
        for model in models:
            if isinstance(model, _SummedEfield):
                self.models.extend(model.models)
            else:
                self.models.append(model)

    def calc_field(self, x, y, z):
        Ex_s, Ey_s, Ez_s = 0, 0, 0
        for model in self.models:
            Ex, Ey, Ez = model.calc_field(x / model.to_meters, y / model.to_meters, z / model.to_meters)
            Ex_s += Ex
            Ey_s += Ey
            Ez_s += Ez
        return Ex_s, Ey_s, Ez_s

    def __str__(self):
        s = self.model_name
        for i, model in enumerate(self.models, 1):
            name, *params = str(model).split("\n")
            s += f"\n\tModel {i}: {name}"
            for param in params:
                s += f"\n\t{param}"
        return s
