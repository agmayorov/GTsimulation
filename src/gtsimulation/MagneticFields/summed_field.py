from datetime import datetime

from MagneticFields import AbsBfield


class Summed(AbsBfield):

    def __init__(self, models, **kwargs):
        super().__init__(**kwargs)
        self.ModelName = "Summary"
        self.models = models

    def CalcBfield(self, x, y, z, **kwargs):
        Bx_s, By_s, Bz_s = 0, 0, 0
        for model in self.models:
            Bx, By, Bz = model.CalcBfield(x / model.ToMeters,
                                          y / model.ToMeters,
                                          z / model.ToMeters)
            Bx_s += Bx
            By_s += By
            Bz_s += Bz
        return Bx_s, By_s, Bz_s

    def UpdateState(self, new_date: datetime):
        for model in self.models:
            if hasattr(model, "UpdateState"):
                model.UpdateState(new_date)

    def to_string(self):
        s = f"""Summed
        #-#-#-#-#-#-#-#-#\n"""
        for model in self.models:
            s += "\t\t"+model.to_string()
            s += "\n\t\t#-#-#-#-#-#-#-#-#\n"
        return s
