import datetime
import importlib

from Global import Units, Regions
from MagneticFields import AbsBfield


class Summary(AbsBfield):
    ToMeters = Units.RE2m

    def __init__(self, date: datetime.datetime, models, **kwargs):
        super().__init__(**kwargs)
        self.Region = Regions.Magnetosphere
        self.ModelName = "Summary"
        self.models = []
        self.Date = date
        module_name = f"MagneticFields.{self.Region.name}"
        module = importlib.import_module(module_name)
        for m in models:
            class_name = m if not isinstance(m, list) else m[0]
            params = {"date": self.Date,
                      **({} if not isinstance(m, list) else m[1])}
            if hasattr(module, class_name):
                B = getattr(module, class_name)
                self.models.append(B(**params))
            else:
                raise Exception("No such field")

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

    def __str__(self):
        s = f"""Summary
        #-#-#-#-#-#-#-#-#\n"""
        for model in self.models:
            s += "\t\t"+model.__str__()
            s += "\n\t\t#-#-#-#-#-#-#-#-#\n"
        return s
