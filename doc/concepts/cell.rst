.. _modelcells:

Cells
=====

The basic unit of abstraction in an Arbor model is a cell.
A cell represents the smallest model that can be simulated.
Cells interact with each other via spike exchange and gap junctions.

.. table:: Identifiers used to uniquely refer to cells and objects like synapses on cells.

    ========================  ================================  ===========================================================
    Identifier                Type                              Description
    ========================  ================================  ===========================================================
    .. generic:: gid          integral                          The unique global identifier of a cell.
    .. generic:: index        integral                          The index of an item in a cell-local collection.
                                                                For example the 7th synapse on a cell.
    .. generic:: cell_member  tuple (:gen:`gid`, :gen:`index`)  The global identification of a cell-local item with `index`
                                                                into a cell-local collection on the cell identified by `gid`.
                                                                For example, the 7th synapse on cell 42.
    ========================  ================================  ===========================================================

Cell interactions via :ref:`connections <modelconnections>` and :ref:`gap junctions <modelgapjunctions>` occur
between **source**, **target** and **gap junction site** locations on a cell. Connections are formed from sources
to targets. Gap junctions are formed between two gap junction sites. An example of a source on a
:ref:`cable cell<model_cable_cell>` is a :ref:`threshold detector <cable-threshold-detectors>` (spike detector);
an example of a target on a :ref:`cable cell<model_cable_cell>` is a :ref:`synapse <cable-synapses>`.

Each cell has a global identifier :gen:`gid`, and each **source**, **target** and **gap junction site** has a
global identifier :gen:`cell_member`. These are used to refer to them in :ref:`recipes <modelrecipe>`.

A cell can have multiple sources, targets and gap junction site objects. Each object is ordered relative to other
objects of the same type on that cell. The unique :gen:`cell_member` (:gen:`gid`, :gen:`index`) identifies an object
according to the :gen:`gid` of the cell it is placed on, and its :gen:`index` on the cell enumerated according to the
order of insertion on the cell relative to other objects of the same type.

The :gen:`gid` of a cell is used to determine its cell :ref:`kind <model_cell_kind>` and
:ref:`description <model_cell_description>` in the :ref:`recipe <modelrecipe>`. The :gen:`cell_member` of a source,
target or gap junction site is used to form :ref:`connections <modelconnections>` and
:ref:`gap junctions <modelgapjunctions>` in the :ref:`recipe <modelrecipe>`.



.. _model_cell_kind:

Cell kind
---------

.. table:: The 4 types of cell supported by Arbor

    ========================  ===========================================================
    Cell Kind                 Description
    ========================  ===========================================================
    **Cable cell**            Cell with morphology described by branching
                              1D cable segments and user configurable dynamics.
    **LIF cell**              Single-compartment no-mechanism leaky integrate-and-fire
                              neuron.
    **Spiking cell**          Proxy cell that generates spikes from a user-supplied
                              time sequence.
    **Benchmark cell**        Proxy cell used for benchmarking (developer use only).
    ========================  ===========================================================

.. _model_cable_cell:
1. **Cable Cells**

   Cable cells are morphologically-detailed cells represented as branching linear 1D segments. They can be coupled
   to other cell types via the following mechanisms:

   1. Spike exchange over a :ref:`connection <modelconnections>` with fixed latency.
      Cable cells can *receive* spikes from any kind of cell, and can be a *source* of spikes
      to cells that have target sites (i.e. *cable* and *lif* cells).
   2. Direct electrical coupling between two cable cells via :ref:`gap junctions <modelgapjunctions>`.

   Key concepts:

   * **Morphology**: The :ref:`morphology <co_morphology>` of a cable cell is composed of a branching
     tree of one-dimensional line segments. Strictly speaking, Arbor represents a morphology as an
     *acyclic directed graph*, with the soma at the root.
   * **Detectors**: Spike :ref:`detectors <cable-threshold-detectors>` generate spikes when the voltage
     at location on the cell passes a threshold. Detectors act as **sources** of
     :ref:`connections <modelconnections>`.
   * **Synapses**: :ref:`Synapses <cable-synapses>` act as **targets** of
     :ref:`connections <modelconnections>`. A synapse is described by a synapse type
     (with associated parameters) and location on a cell.
   * **Gap Junction Sites**: These refer to the :ref:`sites <cable-gj-sites>` of
     :ref:`gap junctions <modelgapjunctions>`. They are declared by specifying a location on a cell.

   Because cable cells are the main cell kind in Arbor and have more properties than listed here,
   they have several :ref:`dedicated pages <cablecell>`.

.. _model_lif_cell:
2. **LIF Cells**

   LIF cells are single compartment leaky integrate and fire neurons with one **source** and one **target**.
   LIF cells do not support adding additional **sources** or **targets**. They do not support **gap junctions**.
   They are typically used to simulate point-neuron networks.

.. _model_spike_cell:
3. **Spiking Cells**

   Spiking cells act as spike sources from values inserted via a `schedule description`.
   They are point neurons with one built-in **source** and no **targets**.
   They do not support adding additional **sources** or **targets**. They do not support **gap junctions**.

.. _model_bench_cell:
4. **Benchmark Cells**

   Benchmark cells are proxy cells used for benchmarking, and used by developers to benchmark the spike exchange and
   event delivery infrastructure.

Most Arbor users will want to use the cable cell, because it is the only cell kind that supports complex
morphologies and user-defined mechanisms. See the cable cell's :ref:`dedicated page <cablecell>` for more info.

.. _model_cell_description:

Cell description
----------------

The `description` of a cell details everything needed to build a cell. This degree of detail needed
differs according to the cell kind.

1. **Cable Cells**

   The description of a cable cell includes the following:

     * :ref:`Morphology <co_morphology>`: the shape of the cell.
     * Discretisation: how to split the morphology into discrete components for the simulation.
     * Initial membrane voltage.
     * Initial axial resistivity.
     * Intial membrane capacitance.
     * Initial temperature.
     * Initial ion internal and external concentrations.
     * Initial ion reversal potential.
     * :ref:`Density mechanisms <mechanisms-density>`: commonly used to describe ion-channel dynamics accross
       :ref:`regions <labels-region>` of the cell.
     * :ref:`Ion reversal potential mechanisms <mechanisms-revpot>`: used to control the reversal potentials of
       ions accross :ref:`regions <labels-region>` of the cell.
     * :ref:`Point mechanisms <mechanisms-point>`: commonly used to describe synapses on specific
       :ref:`locations <labels-locset>` of the cell; typically act as targets in the recipe.
     * Stimuli: such as current clamps; placed on specific :ref:`locations <labels-locset>` on the cell.
     * Threshold detectors: used to generate spiking events on specific :ref:`locations <labels-locset>` on the
       cell, when the voltage increases above a certain threshold; typically act as sources in the recipe.
     * Gap junction sites: used to electrically couple the cell to another gap junction site on another cell;
       placed on specific :ref:`locations <labels-locset>` on the cell.

   The cable cell is the most complex cell kind provided in arbor and is highly customisable. The provided links
   describe each concept in more detail. And the :ref:`C++ <cppcable_cell>` and :ref:`Python <pycable_cell>` APIs
   illustrate how to programmatically provide the cell description in Arbor.

2. **LIF Cells**

   The description of a LIF cell is used to control the leaky integrate-and-fire dynamics.
      * Resting potential.
      * Reset potential.
      * Initial value of membrane potential.
      * Membrane potential decaying constant.
      * Membrane capacitance.
      * Firing threshold.
      * Refractory period.

   The morphology of a lif cell is automatically modeled as a single compartment; each cell has a single built-in
   source and target which do not need to be explicitly added in the cell description.

3. **Spiking cells**

   The description of a spiking cell is used to determine the spiking schedule of the cell. Its morphology is
   automatically modeled as a single compartment; each cell has a single built-in source which does not need to
   be explicitly added in the cell description.

4. **Benchmark Cells**

   The description of a benchmark cell is used to determine the spiking schedule of the cell and manipulate its
   performance efficiency. This cell is mainly used by developers.

API
---

* :ref:`Python <pycell>`
* :ref:`C++ <cppcell>`
