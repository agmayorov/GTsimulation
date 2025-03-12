from abc import ABC, abstractmethod


class GTGeneralMedium(ABC):

    def __init__(self):
        self.region = None
        self.model = None
        self.element_list = []

    @abstractmethod
    def calculate_model(self, x, y, z, date_time, **kwargs):
        pass

    @abstractmethod
    def get_density(self):
        pass

    @abstractmethod
    def get_element_list(self):
        return self.element_list

    @abstractmethod
    def get_element_abundance(self):
        pass

    @abstractmethod
    def to_string(self):
        pass

    def __str__(self):
        return self.to_string()
