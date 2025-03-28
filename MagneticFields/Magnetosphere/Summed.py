import datetime
import importlib

from Global import Units, Regions
from MagneticFields import AbsBfield


class Summed(AbsBfield):
    ToMeters = Units.RE2m

    def __init__(self, date: datetime.datetime, models, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Magnetosphere
        self.ModelName = "Summary"
        self.models = models
        self.Date = date

    def CalcBfield(self, x, y, z, **kwargs):
        Bx_s, By_s, Bz_s = 0, 0, 0
        for model in self.models:
            Bx, By, Bz = model.CalcBfield(x * self.ToMeters / model.ToMeters,
                                          y * self.ToMeters / model.ToMeters,
                                          z * self.ToMeters / model.ToMeters)
            Bx_s += Bx
            By_s += By
            Bz_s += Bz
        return Bx_s, By_s, Bz_s

    def UpdateState(self, new_date: datetime.datetime):
        self.Date = new_date
        for model in self.models:
            model.UpdateState(new_date)

    def to_string(self):
        s = f"""Summed
        #-#-#-#-#-#-#-#-#\n"""
        for model in self.models:
            s += "\t\t"+model.to_string()
            s += "\n\t\t#-#-#-#-#-#-#-#-#\n"
        return s
