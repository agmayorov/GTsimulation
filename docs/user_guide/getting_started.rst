Getting Started
===============

This section demonstrates the basic workflow for running a particle trajectory
simulation with GTsimulation. The example below propagates a single proton in a
uniform magnetic field.

.. note::

   This example is intentionally minimal and demonstrates only the basic
   workflow. More advanced examples, including realistic magnetic field
   models, particle distributions, and interactions with matter, are described
   in the following sections of the User Guide.

Import the package into Python:

.. code-block:: python

   import gtsimulation as gt

Next, create the objects that define the simulation setup. The order in which
these objects are created is generally not important unless they depend on one
another.

As a first step, define a uniform magnetic field with a strength of **100 nT**
directed along the positive *z*-axis:

.. code-block:: python

   b_field = gt.magnetic_field.Uniform(B=[0, 0, 100])

To specify the initial conditions, create a particle flux object. GTsimulation
is primarily designed for Monte Carlo simulations of large particle ensembles.
Therefore, the initial positions, velocities, and energies are defined using
Monte Carlo generators that implement different statistical distributions.

In this example, we simulate a single proton starting at
:math:`(300, 0, 0)\,\mathrm{km}` with an initial velocity directed along
:math:`(0, -1, 1)` and a kinetic energy of **0.1 MeV**.

For convenience, import the particle generator module and the unit system:

.. code-block:: python

   from gtsimulation.particle import generator
   from gtsimulation.common import Units

   particle = gt.particle.Flux(
       Distribution=generator.distribution.UserInput(
           R0=[300 * Units.km, 0, 0],
           V0=[0, -1, 1],
       ),
       Spectrum=generator.spectrum.UserInput(
           energy=0.1 * Units.MeV,
       ),
       Names="proton",
       Nevents=1,
   )

Next, define the integration time step (in seconds) and the number of
integration steps. The latter determines the number of points stored along the
particle trajectory.

.. code-block:: python

   dt = 1e-3  # sec
   n_step = 2000

GTsimulation performs trajectory calculations through simulator classes located
in the :mod:`gtsimulation.pusher` subpackage. Each simulator implements a
particular numerical integration algorithm. In this example, we use the
Buneman–Boris particle pusher.

.. code-block:: python

   simulator = gt.pusher.BunemanBorisSimulator(
       Particles=particle,
       Step=dt,
       Num=n_step,
       Bfield=b_field,
   )

Run the simulation by calling the simulator object:

.. code-block:: python

   result = simulator()

The calculated trajectory can then be extracted from the simulation output and
visualized using Matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt

   track = result[0][0]["Track"]["Coordinates"]

   fig = plt.figure()
   ax = fig.add_subplot(projection="3d")

   ax.plot(track[:, 0], track[:, 1], track[:, 2])

   ax.set_xlabel("x [m]")
   ax.set_ylabel("y [m]")
   ax.set_zlabel("z [m]")

   plt.show()
