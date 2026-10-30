Magnetic field
==============

The magnetic field is implemented through a dedicated **Magnetic Field Module**.
The core of this module is the abstract base class :py:class:`gtsimulation.magnetic_field.AbsBfield`.
Any custom magnetic field model must be defined as a concrete subclass of this class, which requires the explicit
implementation of a :py:meth:`~gtsimulation.magnetic_field.AbsBfield.calc_field` method. This method returns the
magnetic field vector at a given point.

The module comes with a library of pre-implemented models ready for immediate use:

* **General:** A uniform magnetic field.
* **Magnetosphere:** Dipole field, IGRF, Tsyganenko (89, 96, 15), CHAOS (7, 8), CM6, COV-OBS.x2, LCS-1, SIFM, DIFI-6.
* **Heliosphere:** The Parker spiral model.
* **Galaxy:** The Jansson & Farrar (JF12) and Unger & Farrar (UF23) models.

The specific classes that implement these field models can be found within the :py:mod:`gtsimulation.magnetic_field`
module and its submodules. These models support dynamic fields that can evolve in time during particle tracing.
The abstract architecture enables users to seamlessly integrate their own field models by subclassing the base class
and implementing the required interface.

Magnetic fields can be combined with the ``+`` operator. The resulting field is the vector sum of the individual
fields:

.. code-block:: python

   from datetime import datetime
   from gtsimulation.magnetic_field.magnetosphere import Gauss, Tsyganenko

   date = datetime(2008, 1, 1)
   b_field = Gauss(date=date, model="IGRF", model_type="core", version=13) + Tsyganenko(date=date, ModCode="96")

Custom magnetic field
---------------------

A custom magnetic field model is a subclass of :py:class:`~gtsimulation.magnetic_field.AbsBfield` that implements two
methods:

* :py:meth:`~gtsimulation.magnetic_field.AbsBfield.calc_field` takes the coordinates in the native length unit of the
  model and returns the field components in nT. The native unit is set by the class attribute ``to_meters``, the length
  of the unit in metres (for example, ``Units.km``). The default value ``1`` corresponds to metres.
* :py:meth:`~gtsimulation.magnetic_field.AbsBfield.update_state` updates the model for a new date. A model that does
  not depend on time implements it as a no-op.

The example below shows how to implement a custom magnetic field model: a magnetic bottle. The field magnitude on
the :math:`z` axis grows quadratically, :math:`B_z = B_0 \left(1 + z^2 / L^2\right)`, and the radial components
:math:`B_x = -B_0 x z / L^2` and :math:`B_y = -B_0 y z / L^2` keep :math:`\nabla \cdot \mathbf{B} = 0`.
Coordinates are taken in kilometres.

.. code-block:: python

   from gtsimulation.common import Units
   from gtsimulation.magnetic_field import AbsBfield


   class MagneticBottle(AbsBfield):

       to_meters = Units.km  # calc_field takes coordinates in km

       def __init__(self, B0, L):
           super().__init__()
           self.ModelName = "MagneticBottle"
           self.B0 = B0  # field at z = 0, nT
           self.L = L / self.to_meters  # length scale, km

       def calc_field(self, x, y, z):
           Bx = -self.B0 * x * z / self.L ** 2
           By = -self.B0 * y * z / self.L ** 2
           Bz = self.B0 * (1 + (z / self.L) ** 2)
           return Bx, By, Bz

       def update_state(self, new_date):
           pass  # the field does not depend on time

       def __str__(self):
           return (
               f"{self.ModelName}\n"
               f"\tB0: {self.B0} nT\n"
               f"\tL: {self.L} km"
           )

A custom model is used in the same way as the built-in ones:

.. code-block:: python

   import gtsimulation as gt
   from gtsimulation.particle import generator

   particle = gt.particle.Flux(
       Distribution=generator.distribution.UserInput(R0=[0, 0, 0], V0=[1, 0, 1]),
       Spectrum=generator.spectrum.UserInput(energy=1 * Units.keV),
       Names="proton",
       Nevents=1,
   )

   simulator = gt.pusher.BunemanBorisSimulator(
       Particles=particle,
       Step=1e-6,
       Num=20000,
       Bfield=MagneticBottle(B0=1 * Units.mT, L=2 * Units.km),
   )
   result = simulator()
