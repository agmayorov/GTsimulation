Getting started
==========================

Description
------------
**GTsimulation** (GT) is software a package that is created for simulations of propagation of charged particles in electromagnetic fields.
GT solves the relativistic equation of motion of a particle using Buneman-Boris scheme. That allows to recover the trajectory
of a particle with high precision. Additionally, we take into account the energy losses of particles such as, radiation losses
(synchrotron radiation), adiabatic losses (in the heliosphere), and the interactions with the medium. As a result of interaction
with the medium secondary particles may be created, that are later simulated in GT.

The code is written in a flexible manner, and easily can be extended by inheriting from the abstract classes of each module. To
enhance the speed of calculations **numba** just-in-time compiler is used to compile the main functions.


Installing
------------
First update your **python** version into `python3.10.x`. Download the directory form GitHub

.. code-block:: console

    $ git clone https://github.com/agmayorov/GTsimulation.git
    $ cd GTsimulation

Then install the necessary packages for GT

.. code-block:: console

   $ pip install -r requirements

Geant4 Integration
~~~~~~~~~~~~~~~~~~

In order to be able to use **Geant4** features for nuclear interactions, add the **Geant4** path into `path_geant4`
variable in `GTsimulation/Interaction/settings.py`.

.. code-block:: python
    :caption: GTsimulation/Interaction/settings.py

    path_geant4 = "YOUR GEANT4 PATH"


Additionally one needs to build the **Geant4** libraries: *Atmosphere*, *DecayGenerator*, *MatterLayer*. Change the directory
into the directory of the library you want to build (e.g. `GTsimulation/Interaction/G4Source/Atmosphere`) and the *build* it
using **cmake**. Then move the executable into `GTsimulation/Interaction`.

.. code-block:: console

    $ cd GTsimulation/Interaction/G4Source/Atmosphere/build
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make
    $ cp Atmosphere ../../../


Examples
-----------

.. toctree::
   :maxdepth: 4

   Examples.Magnetosphere
   Examples.Heliosphere
   Examples.Galaxy

