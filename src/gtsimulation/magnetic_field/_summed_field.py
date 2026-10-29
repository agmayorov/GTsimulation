import warnings
from datetime import datetime

from gtsimulation.magnetic_field import AbsBfield


class Summed(AbsBfield):

    def __init__(self, models):
        warnings.warn(
            "Summed is deprecated and will be removed in a future version; "
            "combine magnetic fields with '+' instead, e.g. B1 + B2",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__()
        self.ModelName = "Summary"
        self.models = models

    def calc_field(self, x, y, z):
        Bx_s, By_s, Bz_s = 0, 0, 0
        for model in self.models:
            Bx, By, Bz = model.calc_field(x / model.to_meters,
                                          y / model.to_meters,
                                          z / model.to_meters)
            Bx_s += Bx
            By_s += By
            Bz_s += Bz
        return Bx_s, By_s, Bz_s

    def update_state(self, new_date: datetime):
        for model in self.models:
            if hasattr(model, "update_state"):
                model.update_state(new_date)

    def __str__(self):
        s = f"""Summed
        #-#-#-#-#-#-#-#-#\n"""
        for model in self.models:
            s += f"\t\t{model}"
            s += "\n\t\t#-#-#-#-#-#-#-#-#\n"
        return s
