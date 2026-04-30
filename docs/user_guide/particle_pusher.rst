Particle pusher
===============

The particle pusher algorithms are implemented through a dedicated **Algos Module**.
The core of this module is the abstract base class :py:class:`gtsimulation.GTSimulator`.
Any custom integrator must be defined as a concrete subclass of this class, which requires the explicit
implementation of a :py:meth:`~gtsimulation.GTSimulator.AlgoStep` method. This method returns a tuple of three elements:
the updated velocity vector, the new Lorentz factor, and an auxiliary Lorentz factor used in synchrotron radiation
loss calculations

The module comes with a library of pre-implemented algorithms ready for immediate use:

* Runge-Kutta integrators (4th and 6th order).
* Modified Buneman-Boris integrator.
* Vay integrator.
* Higuera-Cary integrator.

The specific classes that implement these pusher algorithms can be found within the :py:mod:`gtsimulation.Algos`
module and its submodules.
The abstract architecture enables users to seamlessly integrate their own particle pushers by subclassing the base class
and implementing the required interface.