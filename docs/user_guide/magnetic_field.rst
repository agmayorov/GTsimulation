Magnetic field
==============

The magnetic field is implemented through a dedicated **Magnetic Field Module**.
The core of this module is the abstract base class :py:class:`gtsimulation.MagneticFields.AbsBfield`.
Any custom magnetic field model must be defined as a concrete subclass of this class, which requires the explicit
implementation of a :py:meth:`~gtsimulation.MagneticFields.AbsBfield.CalcBfield` method. This method returns the
magnetic field vector at a given point.

The module comes with a library of pre-implemented models ready for immediate use:

* **General:** A uniform magnetic field.
* **Magnetosphere:** Earth's Dipole field, IGRF, Tsyganenko (89, 96, 15), CHAOS (7, 8), CM6, COV-OBS.x2, LCS-1, SIFM, DIFI-6.
* **Heliosphere:** The Parker spiral model.
* **Galaxy:** The Jansson & Farrar (JF12) and Unger & Farrar (UF23) models.

The specific classes that implement these field models can be found within the :py:mod:`gtsimulation.MagneticFields`
module and its submodules. These models support dynamic fields that can evolve in time during particle tracing.
The abstract architecture enables users to seamlessly integrate their own field models by subclassing the base class
and implementing the required interface.
