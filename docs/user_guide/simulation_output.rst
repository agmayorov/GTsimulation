Simulation output
=================

Saving files
------------
The files are saved in *.npy* format each saved file contains `Nevents` events in it. This parameter passed to the particle generator,
and defines the number of the particles that are generated. Another parameter regarding the saving is `Nfiles`. It defines how many files
are saved. In total as a result we get `Nfiles`.npy files each contining `Nevents` trajectories. The name of the saved files are 
defined in the `Output` variable.

.. code-block:: python

        Flux = {"Nevents": 200, **other_params}
        Nfiles = 20
        Output = "saving_files_example"

The resulting files we have to following name structure: *saving_files_example_i.npy*, where *i* is the number of the file.

Reading files
-------------
The file is a list of dictionaries (each corresponging to a single event) with following keys: `Track`, `Particle`, `BC`, `Child`, 
and `Additions` (the last one is only present for the :py:mod:`Global.regions.Regions.Magnetosphere`).

.. code-block:: python

        import numpy as np

        events = np.load("reading_files_example_1.npy", allow_pickle=True)
        # len(events) is the number of particles in the file

        # going through each event
        for event in events:
                # event is a dictinoary introduced above
                print(event)

The dictionaries contain the following fields:

1. `Track`: 

- `Coordinates` - Trajectory of a particle

- `Velocities` - The vector of velocity at a given point

- `Efield` - The electric field along the trajectory

- `Bfield` - The magentic field along the trajectory

- `Angle` - The angle of scattering relative to the previous point

- `Path` - The cumulative path of the particle

- `Density` - The cumulative density of the particle alongs its trajectory, i.e. :math:`\int \rho dr`,  where :math:`\rho` is the denisty at the point

- `Clock` - The time pasted from the start

- `Energy` - Kinetic energy along the trajectory

2. `Particle`

- `M` - Mass of the particle

- `Z` - The charge number of the particle

- `PDG` - The pdg code

- `T0` - The initial kinetic energy

- `Gen` - The generation of the particle (primimary, secondary, e.t.c)

3. `BC`

- `WOut` - The code that tells why the simualtion for tha particle has stopped (see :py:mod:`Global.codes.BreakIndex`).

4. `Child`: An array of secondary particles that are born after the interaction. An element of `Child` contains the dictionaries that are described above. 
        
5. `Additions`: The additional parameters that are calculated along the trajectory. See :py:mod:`GT.GT.GTSimulator`

Plotting trajectories
---------------------
To plot the trajectory of the *i*-th particle, we should read `events[i]["Track"]["Coordinates"]` that are in defult **GT** units, i.e. 
in meters (one may want to convert them into another coordinate units, see :py:mod:`Global.consts.Units`). 

.. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt

        # Creating a 3d canvas to plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(f"X, [kpc]")
        ax.set_ylabel(f"Y, [kpc]")
        ax.set_zlabel(f"Z, [kpc]")

        events = np.load("reading_files_example_1.npy", allow_pickle=True)
        # len(events) is the number of particles in the file

        # going through each event
        for event in events:

                # for example converting to [kpc]
                R = event["Track"]["Coordinates"] / Units.kpc
                X, Y, Z = R[:, 0], R[:, 1], R[:, 2]

                ax.plot(X, Y, Z, label=f"The trajectory")
                ax.scatter(X[-1], Y[-1], Z[-1], label="End point", s=30)
                ax.scatter(X[0], Y[0], Z[0], label="Start point", s=30)

        ax.axis('equal') # unscaled axes
        ax.legend()
        plt.show()
