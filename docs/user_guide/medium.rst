Medium
======

The propagation medium is defined through the **Medium Module**. At the heart of this module lies the abstract base
class :py:class:`gtsimulation.medium.GTGeneralMedium`. To create a custom medium model, users must implement a concrete
subclass and define several core methods. The primary method, :py:meth:`~gtsimulation.medium.GTGeneralMedium.calculate_model`,
updates the model's internal state for given coordinates. At each step of the trajectory, the simulator calls it with
the particle coordinates in metres. The subclass must also implement
:py:meth:`~gtsimulation.medium.GTGeneralMedium.get_density` (returns the mass density in kg/m³),
:py:meth:`~gtsimulation.medium.GTGeneralMedium.get_element_list` (returns the list of chemical elements), and
:py:meth:`~gtsimulation.medium.GTGeneralMedium.get_element_abundance` (returns their mass fractions in the same order,
summing to 1).

The module provides several pre-configured models for common environments:

* **General:** A homogeneous medium.
* **Magnetosphere:** The NRLMSIS-00, 2.0, 2.1 atmosphere model.
* **Galaxy:** A model for interstellar gas based on the work of Jóhannesson et al.

The concrete classes for these models are located within the :py:mod:`gtsimulation.medium` module and its submodules.
This module is optional and is only required when simulating physical processes that depend on the medium, such as
nuclear interactions. During trajectory calculation, it tracks the accumulated **grammage** (g/cm²), which is essential
for calculating interaction probabilities. The modular design allows for straightforward integration of user-defined
medium models by subclassing the base class.

Custom medium
-------------

The example below shows how to implement a custom medium model: a hydrogen-helium gas whose density grows
quadratically along the :math:`z` axis, :math:`\rho = \rho_0 \left(1 + z^2 / L^2\right)`. The composition is
constant: 75 % hydrogen and 25 % helium by mass.

.. code-block:: python

   import numpy as np

   from gtsimulation.medium import GTGeneralMedium


   class QuadraticMedium(GTGeneralMedium):

       def __init__(self, rho0, L):
           super().__init__()
           self.model = "QuadraticMedium"
           self.rho0 = rho0  # density at z = 0, kg/m3
           self.L = L  # length scale, m
           self.element_list = ["H", "He"]
           self.element_abundance = np.array([0.75, 0.25])  # mass fractions
           self.density = rho0

       def calculate_model(self, x, y, z):
           self.density = self.rho0 * (1 + (z / self.L) ** 2)

       def get_density(self):
           return self.density

       def get_element_list(self):
           return self.element_list

       def get_element_abundance(self):
           return self.element_abundance

       def __str__(self):
           return (
               f"{self.model}\n"
               f"\trho0: {self.rho0} kg/m3\n"
               f"\tL: {self.L} m"
           )

The medium is passed to the simulator with the ``Medium`` parameter. To save the accumulated grammage along the
trajectory, enable ``Density`` in ``Save``:

.. code-block:: python

   import gtsimulation as gt
   from gtsimulation.common import Units
   from gtsimulation.particle import generator

   particle = gt.particle.Flux(
       Distribution=generator.distribution.UserInput(R0=[0, 0, 0], V0=[1, 0, 1]),
       Spectrum=generator.spectrum.UserInput(energy=1 * Units.keV),
       Names="proton",
       Nevents=1,
   )

   simulator = gt.pusher.BunemanBorisSimulator(
       Particles=particle,
       Step=1e-3,
       Num=2000,
       Bfield=gt.magnetic_field.Uniform(B=[0, 0, 100]),
       Medium=QuadraticMedium(rho0=1e-12, L=1000 * Units.km),
       Save=[1, {"Density": True}],
   )
   result = simulator()

   grammage = result[0][0]["Track"]["Density"]  # g/cm²
