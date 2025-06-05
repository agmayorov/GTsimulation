from gtsimulation.Medium import GTGeneralMedium


class GTUniformMedium(GTGeneralMedium):

    def __init__(self, density = 1., element = 'H'):
        super().__init__()
        self.model = "UniformMedium"
        self.density = density
        self.element_list = [element]
        self.element_abundance = [1]

    def calculate_model(self, *args, **kwargs):
        pass

    def get_density(self):
        return self.density

    def get_element_list(self):
        return self.element_list

    def get_element_abundance(self):
        return self.element_abundance

    def to_string(self):
        return f"""{self.model}
        Density: {self.density} kg/m3"""


class GTVacuum(GTUniformMedium):

    def __init__(self):
        super().__init__(density = 0.)
        self.model = "Vacuum"

    def to_string(self):
        return f"{self.model}"
