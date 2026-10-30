from abc import ABC, abstractmethod

from gtsimulation.common import Regions


class GTGeneralMedium(ABC):
    """
    Abstract base class for medium models.

    At each step of the trajectory, the simulator calls calculate_model with
    the particle coordinates in metres and then reads the result with the
    get_* methods. A subclass stores the result of calculate_model in the
    object: get_density returns the mass density in kg/m3,
    get_element_list returns the chemical symbols of the elements and
    get_element_abundance returns their mass fractions.
    """

    def __init__(self):
        self.region = Regions.Undefined
        self.model = None
        self.element_list = []

    @abstractmethod
    def calculate_model(self, x, y, z):
        """
        Evaluate the model at the given point.

        The result is stored in the object and is returned by the get_* methods.

        Parameters
        ----------
        x, y, z : float
            Coordinates of the point, in metres.
        """
        pass

    @abstractmethod
    def get_density(self):
        """
        Return the mass density at the last evaluated point.

        Returns
        -------
        float
            Mass density, in kg/m3.
        """
        pass

    @abstractmethod
    def get_element_list(self):
        """
        Return the chemical symbols of the elements forming the medium.

        Returns
        -------
        list of str
            Element symbols, such as ``"H"`` or ``"He"``, in the order used by
            get_element_abundance.
        """
        return self.element_list

    @abstractmethod
    def get_element_abundance(self):
        """
        Return the mass fractions of the elements at the last evaluated point.

        Returns
        -------
        array_like of float
            Mass fractions of the elements in the order of get_element_list,
            summing to 1.
        """
        pass
