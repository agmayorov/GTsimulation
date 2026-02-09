# GT simulation

[**GT simulation** (GT)](https://spacephysics.mephi.ru/GTsimulation) is software a package that is created for simulations of propagation of charged particles in electromagnetic fields.
GT solves the relativistic equation of motion of a particle using Buneman-Boris scheme. That allows to recover the trajectory
of a particle with high precision. Additionally, we take into account the energy losses of particles such as, radiation losses
(synchrotron radiation), adiabatic losses (in the heliosphere), and the interactions with the medium. As a result of interaction
with the medium secondary particles may be created, that are later simulated in GT.

The code is written in a flexible manner, and easily can be extended by inheriting from the abstract classes of each module. To
enhance the speed of calculations **numba** just-in-time compiler is used to compile the main functions.


## Installation

GT requires Python 3.10+. To avoid possible package conflicts, you can optionally create an isolated virtual environment
using `venv`:
```console
$ python -m venv gt_env
$ source gt_env/bin/activate
```
See the [official `venv` documentation](https://docs.python.org/3/library/venv.html) for details.

If you plan to use the secondary particle generation functionality, install [Geant4](https://geant4.web.cern.ch/download/11.4.0.html) and activate its environment variables:
```console
$ source /path/to/geant4/bin/geant4.sh
```

Install the package with:
```console
$ pip install gtsimulation
```

If you do not want to use Geant4, specify the build option during installation:
```console
$ pip install --config-settings=cmake.define.BUILD_GEANT4_COMPONENTS=OFF gtsimulation
```

Alternatively, you can install the package from the source repository:
```console
$ git clone --depth 1 https://github.com/agmayorov/GTsimulation.git
$ cd GTsimulation
$ pip install .
```
