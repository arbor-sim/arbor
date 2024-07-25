.. _tutorialtwocellsgapjunction:

Two cells connected via a gap junction
======================================

In this example, we will set up two cells connected via a gap junction. Each of
the cells has a passive leak current as its only dynamics. This plus the gap
junction will produce an equilibrium potential different from both the resting
potentials. We will investigate how the equilibrium potentials of the two cells
change due to the gap junction connection.

.. figure:: network_two_cells_gap_junctions_circuit.svg
    :width: 400
    :align: center

    The equivalent circuit used to calculate the equilibrium potentials.

.. Note::

   **Concepts covered in this example:**

   1. Creating a simulation recipe for two cells.
   2. Placing probes.
   3. Running the simulation and extracting the results.
   4. Adding a gap junction connection.

We assume prior exposure to the concepts of cable cells, recipes, and simple
networks.

Walk-through
************

We set up a recipe for the simulation of two cells

.. literalinclude:: ../../python/example/network_two_cells_gap_junctions.py
   :language: python
   :lines: 13-37

in which we store the relevant parameters for the two cells, all of which are
shared except the equilibrium potentials in ``Vms``. These are used to build
the network.

Let's quickly check some standard callbacks:

- ``num_cells`` returns the number of cells in the network, fixed as 2
- ``cell_kind`` specifies that we handle ``cable_cell`` exclusively.
- ``global_properties`` returns a list of standard parameters based on the
  defaults of the NEURON simulator.
- ``probes`` record the membrane potential at the cell mid.

The two remaining methods are:

``cell_description``
--------------------

We construct a basic, single-segment morphology from the ``length`` and
``radius`` parameters. The decor sets the basic parameters and adds the passive
leak current ``pas`` with the given resting value ``Vms[gid]`` and conductivity
``g``.

.. literalinclude:: ../../python/example/network_two_cells_gap_junctions.py
   :language: python
   :lines: 58

The only new item is the placement of the gap junction endpoint at ``midpoint``
with the basic, builtin ``gj`` dynamics type (other dynamics may be defined and
used).


``gap_junctions_on``
-------------------

Similar to ``connections_on``, this method returns a list of gap junction
connections and these are defined in the same manner.

.. literalinclude:: ../../python/example/network_two_cells_gap_junctions.py
   :language: python
   :lines: 69-70

By ``(gid + 1) % 2`` we define two connections, one per cell, between the cells.
This is due to the uni-directional definition of gap junctions in Arbor.

Running the simulation
**********************

To allow runtime configuration, we define a parser for command line arguments
which are used to set parameters in the recipe

.. literalinclude:: ../../python/example/network_two_cells_gap_junctions.py
   :language: python
   :lines: 76-98


We then set up the simulation and configure a sampling width equal to the timestep
:math:`dt`. Now, we can run the network.

.. literalinclude:: ../../python/example/network_two_cells_gap_junctions.py
   :language: python
   :lines: 100-113

All that is left to do is to put this into a plot. The output plot below shows
how the potential of the two cells approaches their equilibrium potentials, which
can be computed from the given parameters.

.. math::

   \bar U_i = U_i + w(U_i - U_j)

   w = \frac{\rho + g}{2 \rho + g}

.. figure:: network_two_cells_gap_junctions_result.svg
    :width: 800
    :align: center


The full code
**************

You can find the full code of the example at ``python/examples/network_two_cells_gap_junctions.py``

Executing the script will run the simulation with default parameters.
