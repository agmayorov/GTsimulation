Simulation input
================

Before running a simulation, you need to define the initial conditions of the particles: their initial coordinates,
direction of motion, energy, particle type, and the total number of particles. You can either specify these quantities
directly or use Monte Carlo generators to sample them from predefined distributions. GT provides
the :mod:`gtsimulation.particle` subpackage for defining and generating initial particle conditions.

Coordinate system and reference frame
-------------------------------------
GT uses a Cartesian coordinate system. Its orientation depends on the problem you are studying and is generally
determined by the magnetic-field model used in the simulation. All models in a simulation must use the same coordinate
system and reference frame. In particular, the simulation interprets the geometry of the medium, magnetic and electric
fields, as well as the initial particle coordinates and directions of motion, within the same reference frame.

GT introduces the concept of a *region* to group models associated with near-Earth space, the heliosphere, and the
Galaxy, and to facilitate the implementation of region-specific functionality. The pre-implemented models in each
region use a specific reference frame:

- In the magnetosphere, GT uses the Earth-centered, Earth-fixed coordinate system (ECEF) [1]_, also known as the
  geocentric coordinate system or Geographic (GEO) coordinate system [2]_.
- In the heliosphere, GT uses the Heliocentric Earth Ecliptic (HEE) coordinate system [2]_, [3]_.
- In the Galaxy, GT uses a Galactocentric coordinate system with the `x`-axis oriented from the Sun toward the Galactic
  center, following the convention used in [4]_.

.. [1] https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system
.. [2] Hapgood, M. A. (1992). Space physics coordinate transformations: A user guide. Planetary and Space Science, 40(5), 711-717.
.. [3] https://en.wikipedia.org/wiki/Solar_coordinate_systems#Heliocentric
.. [4] Jansson, R., & Farrar, G. R. (2012). A new model of the galactic magnetic field. The Astrophysical Journal, 757(1), 14.

Units
-----
GT performs all calculations using seconds, meters, MeV, and nT. You therefore need to provide initial coordinates and
particle energies in these units, regardless of the problem you are studying.

For convenience, GT provides the :class:`gtsimulation.common.Units` class with conversion factors for units such
as ``km``, ``AU``, ``pc``, ``kpc``, ``GeV``, and ``TeV``. You can use these attributes directly in expressions, for
example, ``1.5 * Units.AU`` instead of ``2.244e11``. This approach was inspired by and adopted from Geant4.

When you specify an initial particle velocity explicitly, GT uses it only to determine the direction of motion.
The particle energy determines the magnitude of the velocity.

GT stores the resulting particle trajectories as sets of coordinates in meters, expressed in the simulation reference
frame.

Initial conditions and particle flux
------------------------------------
The :class:`gtsimulation.particle.Flux` class provides the main interface for defining particle initial conditions in
the :mod:`gtsimulation.particle` subpackage. You can use it when the spatial and energy distributions of the initial
particles are independent of each other and of other aspects of the simulation, such as the magnetic field.
The :class:`~gtsimulation.particle.Flux` class therefore takes a spatial distribution generator, an energy distribution
generator, a list of particle types, and the total number of particles.

The :mod:`gtsimulation.particle.generator` module contains submodules with classes for generating
spatial (:mod:`~gtsimulation.particle.generator.distribution`) and
energy (:mod:`~gtsimulation.particle.generator.spectrum`) distributions.
