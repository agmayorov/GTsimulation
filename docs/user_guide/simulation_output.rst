Simulation output
=================

Saving files
------------
The files are saved in **.npy** format. Each file contains `Nevents` events (trajectories). The `Nevents` parameter is
passed to the particle generator and defines how many particles are generated per file.

Another parameter related to saving is `Nfiles`, which defines how many files are created. In total, `Nfiles` files will
be saved, each containing `Nevents` trajectories. The base name for the saved files is defined by the `Output` variable.

.. code-block:: python

    Flux = {"Nevents": 200, **other_params}
    Nfiles = 20
    Output = "saving_files_example"

The resulting files follow the naming structure: ``saving_files_example_i.npy``, where ``i`` is the file index (starting from 1).

Reading files
-------------
Each file contains a list of dictionaries, where each dictionary corresponds to a single particle event. The dictionary
has the following keys:

- `Track`
- `Particle`
- `BC`
- `Child`
- `Additions` (only present for the :py:attr:`~gtsimulation.Global.regions.Regions.Magnetosphere` region)

.. code-block:: python

    import numpy as np

    events = np.load("reading_files_example_1.npy", allow_pickle=True)
    # len(events) equals the number of particles in the file

    # Iterate over each event
    for event in events:
        # event is a dictionary with the structure described above
        print(event)

Dictionary structure
^^^^^^^^^^^^^^^^^^^^

1. **Track** – contains trajectory data:

   - `Coordinates` – particle trajectory points
   - `Velocities` – velocity vectors at each point
   - `Efield` – electric field along the trajectory
   - `Bfield` – magnetic field along the trajectory
   - `Angle` – scattering angle relative to the previous point
   - `Path` – cumulative path length
   - `Density` – amount of matter traversed: :math:`\int \rho \, dr`, where :math:`\rho` is the density at each point
   - `Clock` – elapsed time from the start
   - `Energy` – kinetic energy along the trajectory

2. **Particle** – particle properties:

   - `M` – mass
   - `Z` – charge number
   - `PDG` – PDG code
   - `T0` – initial kinetic energy
   - `Gen` – particle generation (primary, secondary, etc.)

3. **BC** – break conditions:

   - `WOut` – break code indicating why the simulation stopped (see :py:data:`~gtsimulation.Global.codes.BreakCode`)

4. **Child** – array of secondary particles produced in interactions. Each element has the same structure as described above.

5. **Additions** – additional calculated parameters (see :py:class:`~gtsimulation.GTSimulator` class for details).

Plotting trajectories
---------------------
To plot the trajectory of the *i*-th particle, use ``events[i]["Track"]["Coordinates"]``. Coordinates are in
the default **GT** units (meters). You may convert them to other units using the :py:data:`gtsimulation.Global.consts.Units` module.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from Global.consts import Units

    # Load data
    events = np.load("reading_files_example_1.npy", allow_pickle=True)

    # Plot only the first event for clarity
    event = events[0]

    # Convert to desired units (e.g., kpc)
    R = event["Track"]["Coordinates"] / Units.kpc
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_zlabel("Z [kpc]")

    # Plot trajectory and markers
    ax.plot(X, Y, Z, label="Trajectory")
    ax.scatter(X[0], Y[0], Z[0], label="Start point")
    ax.scatter(X[-1], Y[-1], Z[-1], label="End point")

    ax.legend()
    plt.show()
