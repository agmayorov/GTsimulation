Medium
======

The propagation medium is defined through the **Medium Module**. At the heart of this module lies the abstract base
class :py:class:`gtsimulation.Medium.GTGeneralMedium`. To create a custom medium model, users must implement a concrete
subclass and define several core methods. The primary method, :py:meth:`~gtsimulation.Medium.GTGeneralMedium.calculate_model`,
updates the model's internal state for given coordinates. The subclass must also implement
:py:meth:`~gtsimulation.Medium.GTGeneralMedium.get_density` (returns the mass density),
:py:meth:`~gtsimulation.Medium.GTGeneralMedium.get_element_list` (returns the list of chemical elements), and
:py:meth:`~gtsimulation.Medium.GTGeneralMedium.get_element_abundance` (returns their fractional abundances).

The module provides several pre-configured models for common environments:

* **General:** A homogeneous medium.
* **Magnetosphere:** The NRLMSIS-00, 2.0, 2.1 atmosphere model.
* **Galaxy:** A model for interstellar gas based on the work of Jóhannesson et al.

The concrete classes for these models are located within the :py:mod:`gtsimulation.Medium` module and its submodules.
This module is optional and is only required when simulating physical processes that depend on the medium, such as
nuclear interactions. During trajectory calculation, it tracks the accumulated **grammage** (g/cm²), which is essential
for calculating interaction probabilities. The modular design allows for straightforward integration of user-defined
medium models by subclassing the base class.
