import multiprocessing as mp
import numpy as np
from ._build_config import GEANT4_COMPONENTS_AVAILABLE


PRIMARY_DTYPE = np.dtype({
    'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'KineticEnergy', 'MomentumDirection', 'Position', 'LastProcess'],
    'formats': ['U16', 'i4', 'f8', 'i4', 'f8', '(3,)f8', '(3,)f8', 'U32']
})

SECONDARY_DTYPE = np.dtype({
    'names': ['Name', 'PDGcode', 'Mass', 'Charge', 'KineticEnergy', 'MomentumDirection'],
    'formats': ['U16', 'i4', 'f8', 'i4', 'f8', '(3,)f8']
})


def convert_to_numpy(run_result):
    """
    Convert C++ Geant4 result object to structured NumPy arrays.

    Parameters
    ----------
    run_result : matter_layer.RunResult
        C++ result object returned by the Geant4 simulator.

    Returns
    -------
    primary : numpy.ndarray
        Structured array of shape ``(1,)`` with dtype ``PRIMARY_DTYPE`` containing
        the final state of the primary particle.
    secondary : numpy.ndarray
        Structured array of shape ``(N,)`` with dtype ``SECONDARY_DTYPE`` containing
        secondary particles produced during the interaction (empty if ``N=0``).

    Notes
    -----
    This is an internal utility function used by the multiprocessing worker to ensure
    efficient serialization of complex C++ objects across process boundaries.
    """
    p = run_result.primary
    primary_arr = np.array((
        p.name, p.pdgCode, p.mass, p.charge, p.kineticEnergy,
        np.array(p.momentumDirection), np.array(p.position), p.lastProcess
    ), dtype=PRIMARY_DTYPE)
    secondaries_list = [
        (s.name, s.pdgCode, s.mass, s.charge, s.kineticEnergy, np.array(s.momentumDirection))
        for s in run_result.secondaries
    ]
    secondary_arr = np.array(secondaries_list, dtype=SECONDARY_DTYPE) \
        if secondaries_list else np.empty(0, dtype=SECONDARY_DTYPE)
    return primary_arr, secondary_arr


def sim_worker(input_queue, output_queue, seed):
    """
    Geant4 worker process loop.

    Runs in an isolated process and owns a single ``matter_layer.Simulator`` instance.
    Receives simulation parameters via ``input_queue``, executes events, and returns
    results via ``output_queue``. Automatically terminates when receiving ``None``.

    Parameters
    ----------
    input_queue : multiprocessing.Queue
        Input queue containing parameter tuples: ``(pdg, energy, mass, density, el_names, el_fracs)``.
    output_queue : multiprocessing.Queue
        Output queue for results: ``(primary_array, secondary_array, material_count)``.
    seed : int
        Random seed passed to the Geant4 simulator constructor.

    Notes
    -----
    Internal worker function for ``NuclearInteraction``. Not intended for direct use.
    The worker process is restarted periodically to control Geant4 memory growth.
    """
    from . import matter_layer
    sim = matter_layer.Simulator(seed=seed)
    while True:
        params = input_queue.get()
        if params is None: break
        run_result = sim.run(*params)
        count = sim.material_count
        output_queue.put((*convert_to_numpy(run_result), count))


class NuclearInteraction:
    """
    Geant4-backed nuclear interaction simulator with optional process restarts.

    This class wraps a Geant4 simulation that propagates a single charged particle through
    a homogeneous material layer and returns the final primary particle state and the list
    of produced secondary particles.

    The simulation is executed in an isolated ``multiprocessing.Process``. This makes it
    possible to fully reset Geant4 state by restarting the worker process when needed.

    Performance note: restarting the worker is mainly useful when the medium is
    frequently updated (e.g., many unique material compositions/densities over time).
    In that scenario, Geant4 internal tables may grow and recalculations for previously
    used (now irrelevant) materials can slow down subsequent runs, so periodic restarts
    can keep performance stable.

    If you keep using the same material configuration for many runs, restarts are usually
    unnecessary; the worker can stay alive and repeated calls are significantly faster.

    The target geometry is a cylinder filled with a user-defined material mixture:

    - Cylinder length is computed as ``thickness = mass / density / 1e2`` [m].
    - Cylinder radius equals its length.
    - The primary particle starts at (0, 0, 0) and travels along the +Z axis.
    - Tracking stops when the primary particle dies or reaches the cylinder boundary.

    Internally, the C++ result object is converted to structured NumPy arrays with the
    dtypes ``PRIMARY_DTYPE`` and ``SECONDARY_DTYPE``.

    Parameters
    ----------
    max_generations : int, default=1
        Maximum number of secondary particle generations to model in the simulation.
    grammage_threshold : float, default=10.
        Grammage threshold [g/cm²] above which the Geant4 subroutine is triggered.
        Should be set as a fraction of the expected nuclear interaction length in the material.
    seed : int
        Random seed used to initialize the Geant4 simulator inside the worker process.
    restart_limit : int, default=20
        Number of runs after which the worker process is restarted automatically.
    """

    def __init__(
            self,
            max_generations: int = 1,
            grammage_threshold: float = 10.,
            seed: int = None,
            restart_limit: int = 20
    ):
        if not GEANT4_COMPONENTS_AVAILABLE:
            raise ValueError(
                "GTsimulation was installed without Geant4 support. "
                "Please reinstall the package or disable nuclear interactions."
            )
        self.max_generations = max_generations
        self.grammage_threshold = grammage_threshold  # g/cm2
        self.seed = np.random.randint(2147483647) if seed is None else seed
        self.restart_limit = restart_limit
        self.restart_counter = 0
        self.process = None
        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.__start_new_process()

    def __start_new_process(self):
        if self.process: self.__terminate_process()
        self.restart_counter = 0
        self.process = mp.Process(target=sim_worker, args=(self.in_q, self.out_q, self.seed))
        self.process.daemon = True
        self.process.start()

    def __terminate_process(self):
        if self.process and self.process.is_alive():
            self.in_q.put(None)
            self.process.join(timeout=1)
            if self.process.is_alive(): self.process.terminate()
        self.process = None

    def run_matter_layer(
            self,
            pdg: int,
            energy: float,
            mass: float,
            density: float,
            element_name: list[str],
            element_abundance: list[float]
    ):
        """
        Simulate interaction of a charged particle with a homogeneous material layer.

        Parameters
        ----------
        pdg : int
            PDG code of the primary particle.
        energy : float
            Primary particle kinetic energy in MeV.
        mass : float
            Traversed mass thickness in g/cm^2.
        density : float
            Medium density in g/cm^3.
        element_name : list of str
            Chemical element symbols (e.g. ``["N", "O"]``) forming the medium.
        element_abundance : list of float
            Mass fractions (or the fractions expected by your Geant4 material definition);
            the sum should be 1.

        Returns
        -------
        primary : numpy.ndarray
            Structured NumPy array of shape ``(1,)`` with dtype ``PRIMARY_DTYPE``.
            Fields:

            - ``Name`` : str
            - ``PDGcode`` : int
            - ``Mass`` : float, MeV
            - ``Charge`` : int
            - ``KineticEnergy`` : float, MeV
            - ``MomentumDirection`` : (3,) float, unit vector
            - ``Position`` : (3,) float, m
            - ``LastProcess`` : str

        secondary : numpy.ndarray
            Structured NumPy array with dtype ``SECONDARY_DTYPE`` and shape ``(N,)``,
            where ``N`` is the number of secondary particles (may be 0).
            Fields:

            - ``Name`` : str
            - ``PDGcode`` : int
            - ``Mass`` : float, MeV
            - ``Charge`` : int
            - ``KineticEnergy`` : float, MeV
            - ``MomentumDirection`` : (3,) float, unit vector
        """
        if self.restart_counter >= self.restart_limit:
            self.__start_new_process()
        self.in_q.put((pdg, energy, mass, density, element_name, element_abundance))
        primary, secondary, self.restart_counter = self.out_q.get()
        return primary, secondary
