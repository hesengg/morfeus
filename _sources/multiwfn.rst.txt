==================
Multiwfn Interface
==================

Morfeus provides an interface to the `Multiwfn program`_ for wavefunction analysis as described by Tian Lu :footcite:`lu_2024` :footcite:`lu_chen_2012`.

Multiwfn itself must be installed separately and available as ``Multiwfn`` in
your shell environment. Refer to the `Multiwfn manual`_ for installation instructions.

The optional dependency ``pexpect`` is required to run any function in this module.

Caution: This module has only been tested with molden files derived from xTB, PySCF and ORCA.
The `molden2aim`_ utility is recommended to standardize wavefunction files.

******
Module
******

The :py:class:`Multiwfn <morfeus.multiwfn.Multiwfn>` class allows calculation
of a variety of descriptors and properties from wavefunction (molden/wfn)
files.
Additionally, grid files (``.cub`` or ``.grd``) can be generated or integrated.

Several descriptors are spin-dependent and are handled by
``has_spin=<bool|None>``, detected automatically if not provided.

All available options can be easily retrieved with the ``list_options()`` method.

Citations used by the selected analyses can be collected directly:

.. code-block:: python
  :caption: Example

  >>> mwfn = Multiwfn("molden.input", run_path="mwfn_run", has_spin=False)
  >>> mwfn.get_charges("hirshfeld")
  >>> mwfn.get_citations()


#######################
Charges and Bond Orders
#######################

.. code-block:: python
  :caption: Example

  >>> from morfeus import Multiwfn
  >>> mwfn = Multiwfn("molden.input", run_path="mwfn_run")
  >>> options = mwfn.list_options()
  >>> sorted(options.keys())
  ['bond_order', 'charges', 'descriptors', 'descriptors_fast', 'grid_quality', 'surface']

  >>> charges = mwfn.get_charges(model="adch")
  >>> surface = mwfn.get_surface(model="esp")
  >>> bond_orders = mwfn.get_bond_order(model="mayer")
  >>> sorted(surface.keys())
  ['atomic', 'global']

#######################
Fuzzy-space descriptors
#######################

Atomic descriptors can be calculated for a variety of real-space functions.
The function names can be listed with
``mwfn.list_options()["descriptors"]``. A sublist excluding costly functions is
given as ``mwfn.list_options()["descriptors_fast"]``.

The descriptors can be calculated with:

.. code-block:: python
  :caption: Example

  >>> mwfn = Multiwfn("molden.input", run_path="mwfn_run")
  >>> rho = mwfn.get_descriptor("rho")

Caution: Some functions may not be available or meaningful for some
wavefunction types or spin states.

##########
Grid files
##########

Grid files can be generated for all descriptors with ``get_grid()``
(grid quality significantly affects the computational cost and accuracy).
Cube files can directly be integrated per atom with ``grid_to_descriptors()``.

.. code-block:: python
  :caption: Example

  >>> mwfn = Multiwfn("molden.input", run_path="mwfn_run")
  >>> cube_path = mwfn.get_grid("rho", "low", grid_file_name="rho.cub")
  >>> cube_path.name
  'rho.cub'
  >>> integrated = mwfn.grid_to_descriptors(cube_path)

#######################
Conceptual DFT analyses
#######################

These analyses require a closed-shell wavefunction, so set ``has_spin=False``
or let spin be auto-detected for a closed-shell input.

.. code-block:: python
  :caption: Example

  >>> mwfn = Multiwfn("molden.input", run_path="mwfn_run", has_spin=False)
  >>> fukui = mwfn.get_fukui()
  >>> superdeloc = mwfn.get_superdelocalizabilities()


**********
Background
**********

Morfeus controls Multiwfn's interactive menu system with scripted command
sequences. Results are parsed into Python dictionaries indexed by atom number
(1-based indexing) or descriptor names.

For Multiwfn setup and menu-level details, see the `Multiwfn manual`_.

.. footbibliography::

.. _Multiwfn program: http://sobereva.com/multiwfn/
.. _Multiwfn manual: http://sobereva.com/multiwfn/Multiwfn_manual.html
.. _molden2aim: https://github.com/zorkzou/Molden2AIM