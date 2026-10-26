Simulation output
=================

Saving files
------------

The output format is selected by the extension of the ``Output`` file name:

- ``.npy`` saves the legacy NumPy representation;
- ``.h5`` or ``.hdf5`` saves an HDF5 file.

The extension is required. If ``Output`` is ``None``, no file is created and
the simulator returns the results in memory.

Each file contains ``Nevents`` events (trajectories). ``Nevents`` is passed to
the particle generator and defines how many particles are generated per file.
``Nfiles`` defines how many files are created.

.. code-block:: python

    Flux = {"Nevents": 200, **other_params}
    Nfiles = 20

    # Legacy NumPy output
    Output = "simulation.npy"

    # HDF5 output
    Output = "simulation.h5"

With ``Nfiles = 1``, these examples create ``simulation.npy`` or
``simulation.h5``. With an integer ``Nfiles`` greater than one, the file index
is inserted before the extension, for example ``simulation_0.h5``,
``simulation_1.h5``, and so on.

GT also writes the simulation parameters to ``simulation_params.json``.

HDF5 options
^^^^^^^^^^^^

HDF5 writes trajectory points incrementally instead of keeping the complete
trajectory in memory. Its options can be supplied together with the file name:

.. code-block:: python

    Output = [
        "simulation.h5",
        {
            "ChunkSize": 4096,
            "Compression": None,
        },
    ]

``ChunkSize`` is the number of saved trajectory points buffered before they
are written. Its default value is 4096. ``Compression`` can be ``None``,
``"lzf"``, or ``"gzip"``. Compression can reduce the file size at the cost of
additional processing time.

.. note::

   When ``TrackParams`` or ``ParticleOrigin`` is enabled, GT currently keeps
   the complete event trajectory in memory because the additional parameters
   are calculated after trajectory propagation. HDF5 output remains
   functional in this mode, but it does not provide the same reduction in
   memory consumption.

Reading NumPy files
-------------------

A NumPy output file contains an object array with one result dictionary per
event:

.. code-block:: python

    import numpy as np

    events = np.load("simulation_0.npy", allow_pickle=True)

    for event in events:
        print(event)


Reading HDF5 files
------------------

An HDF5 file stores events in separate groups:

.. code-block:: text

    events/
        0/
            Track/
                Coordinates
                Velocities
                ...
            Particle/
            BC/
            Child/
        1/
            ...

The trajectory datasets can be sliced without loading the complete trajectory
into memory:

.. code-block:: python

    import h5py

    with h5py.File("simulation_0.h5", "r") as file:
        event = file["events"]["0"]

        # Read the complete coordinate array.
        coordinates = event["Track"]["Coordinates"][:]

        # Or read only the first 100,000 saved points.
        coordinates_part = event["Track"]["Coordinates"][:100_000]

        # Scalar datasets are read with [()].
        pdg = event["Particle"]["PDG"][()]

HDF5 files and event groups have a ``complete`` attribute. It is set to
``True`` only after the corresponding calculation has finished successfully:

.. code-block:: python

    with h5py.File("simulation_0.h5", "r") as file:
        if not file.attrs["complete"]:
            print("The file contains an incomplete calculation")

        for event_id, event in file["events"].items():
            if event.attrs["complete"]:
                print(f"Event {event_id} can be analysed")

If a calculation stops with an exception, successfully completed events can
therefore be distinguished from a partially written event. This does not
guarantee that a file remains readable after an operating-system failure,
power loss, or storage failure.

Event structure
---------------

Each event has the same logical sections in NumPy and HDF5 output:

- ``Track`` -- trajectory data;
- ``Particle`` -- particle properties;
- ``BC`` -- the condition which stopped propagation;
- ``Child`` -- secondary particles produced in interactions;
- ``Additions`` -- optional region-specific parameters.

``Track`` can contain the following saved quantities:

- ``Coordinates`` -- particle trajectory points;
- ``Velocities`` -- velocity directions at the saved points;
- ``Efield`` -- electric field along the trajectory;
- ``Bfield`` -- magnetic field along the trajectory;
- ``Angles`` -- angle relative to the previous trajectory point;
- ``Path`` -- cumulative path length;
- ``Density`` -- amount of matter traversed,
  :math:`\int \rho\,dr`, where :math:`\rho` is the density;
- ``Clock`` -- elapsed time from the start;
- ``Energy`` -- kinetic energy along the trajectory;
- ``PitchAngles`` -- particle pitch angle;
- ``LarmorRadii`` -- particle Larmor radius;
- ``GuidingCenter`` -- guiding-centre coordinates.

The exact contents of ``Track`` depend on the ``Save`` configuration.

``Particle`` contains the particle PDG code, mass, charge, generation, initial
coordinates, initial velocity direction, and initial kinetic energy. ``BC``
contains ``WOut``, the break code indicating why propagation stopped; see
:py:data:`~gtsimulation.common.codes.BreakCode`.

Plotting trajectories
---------------------

The following example plots the first trajectory from an HDF5 file. Coordinates
are stored in the default GT units (metres) and can be converted with
:py:data:`~gtsimulation.common.Units`.

.. code-block:: python

    import h5py
    import matplotlib.pyplot as plt
    from gtsimulation.common import Units

    with h5py.File("simulation_0.h5", "r") as file:
        coordinates = file["events"]["0"]["Track"]["Coordinates"][:]

    coordinates = coordinates / Units.kpc
    x, y, z = coordinates.T

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    axes.set_xlabel("X [kpc]")
    axes.set_ylabel("Y [kpc]")
    axes.set_zlabel("Z [kpc]")

    axes.plot(x, y, z, label="Trajectory")
    axes.scatter(x[0], y[0], z[0], label="Start point")
    axes.scatter(x[-1], y[-1], z[-1], label="End point")
    axes.legend()

    plt.show()
