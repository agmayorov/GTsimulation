from Medium import GTGeneralMedium


class GTVacuum(GTGeneralMedium):

    def __init__(self):
        super().__init__()
        self.model = "Vacuum"
        self.density = 0
        self.element_abundance = None

    def calculate_model(self, x, y, z, date_time, **kwargs):
        pass

    def get_density(self):
        return self.density

    def get_element_abundance(self):
        return self.element_abundance
