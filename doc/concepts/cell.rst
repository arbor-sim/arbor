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
    ========================  ================================  ===========================================================

Cell interactions via :ref:`connections <modelconnections>` and :ref:`gap junctions <modelgapjunctions>` occur
between **source**, **target** and **gap junction site** locations on a cell. Connections are formed from sources
to targets. Gap junctions are formed between two gap junction sites. An example of a source on a
:ref:`cable cell<modelcablecell>` is a :ref:`threshold detector <cablecell-threshold-detectors>` (spike detector);
an example of a target on a cable cell is a :ref:`synapse <cablecell-synapses>`.

Each cell has a global identifier :gen:`gid`, and each **source**, **target** and **gap junction site** has a
global identifier :gen:`cell_member`. These are used to refer to them in :ref:`recipes <modelrecipe>`.

A cell can have multiple sources, targets and gap junction site objects. Each object is ordered relative to other
objects of the same type on that cell. The unique :gen:`cell_member` (:gen:`gid`, :gen:`index`) identifies an object
according to the :gen:`gid` of the cell it is placed on, and its :gen:`index` on the cell enumerated according to the
order of insertion on the cell relative to other objects of the same type.

The :gen:`gid` of a cell is used to determine its cell :ref:`kind <modelcellkind>` and
:ref:`description <modelcelldesc>` in the :ref:`recipe <modelrecipe>`. The :gen:`cell_member` of a source,
target or gap junction site is used to form :ref:`connections <modelconnections>` and
:ref:`gap junctions <modelgapjunctions>` in the recipe.



.. _modelcellkind:

Cell kind
---------

.. table:: The 4 types of cell supported by Arbor

    ========================  ===========================================================
    Cell Kind                 Description
    ========================  ===========================================================
    **Cable cell**            Cell with morphology and user configurable dynamics.
    **LIF cell**              Leaky integrate-and-fire neuron.
    **Spiking cell**          Proxy cell that generates spikes.
    **Benchmark cell**        Proxy cell used for benchmarking (developer use only).
    ========================  ===========================================================

.. _modelcablecell:

1. **Cable Cells**

   Cable cells are morphologically-detailed cells. They can be coupled to other cells via the following
   mechanisms:

   1. Spike exchange over a :ref:`connection <modelconnections>` with fixed latency.
      Cable cells can *receive* spikes from any kind of cell, and can be a *source* of spikes
      to cells that have target sites (i.e. *cable* and *lif* cells).
   2. Direct electrical coupling between two cable cells via :ref:`gap junctions <modelgapjunctions>`.

.. _modellifcell:

2. **LIF Cells**

   LIF cells are single-compartment leaky integrate and fire neurons. They are typically used to simulate
   point-neuron networks.

   LIF cells can only interact with other cells via spike exchange over a
   :ref:`connection <modelconnections>` where they can receive spikes from any kind of cell, and can
   be a *source* of spikes to cells that have target sites (i.e. *cable* and *lif* cells).

.. _modelspikecell:

3. **Spiking Cells**

   Spiking cells act as spike sources from user-specified values inserted via a `schedule description`.
   They are typically used as stimuli in a network of more complex cells.

   Spiking Cells can only interact with other cells via spike exchange over a
   :ref:`connection <modelconnections>` where they be a *source* of spikes to cells that have target sites
   (i.e. *cable* and *lif* cells), but they can not *receive* spikes.

.. _modelbenchcell:

4. **Benchmark Cells**

   Benchmark cells are proxy cells used for benchmarking, and used by developers to benchmark the spike
   exchange and event delivery infrastructure.

.. _modelcelldesc:

Cell description
----------------

The `description` of a cell is referred to in the :ref:`recipe <modelrecipe>`, and elsewhere in the docs.
It details everything needed to build a cell. The degree of detail differs according to the cell kind.

1. **Cable Cells**

   The description of a cable cell can include all the following:

     * :ref:`Morphology <morph>`: composed of a branching tree of one-dimensional line segments.
       Strictly speaking, Arbor represents a morphology as an *acyclic directed graph*, with the soma at
       the root.
     * Discretisation: specifies how to split the morphology into discrete components for the simulation.
     * Initial membrane voltage.
     * Initial axial resistivity.
     * Intial membrane capacitance.
     * Initial temperature.
     * Initial ion internal and external concentrations.
     * Initial ion reversal potential.
     * Stimuli: such as current clamps; placed on specific :ref:`locations <labels-locset>` on the cell.
     * :ref:`Density mechanisms <mechanisms-density>`: commonly used to describe ion-channel dynamics across
       :ref:`regions <labels-region>` of the cell.
     * :ref:`Ion reversal potential mechanisms <mechanisms-revpot>`: used to control the reversal potentials of
       ions across regions of the cell.
     * Synapses: implemented using :ref:`point mechanisms <mechanisms-point>` on specific locations of the cell;
       typically act as **targets** of :ref:`connections <modelconnections>` in the recipe.
     * Detectors: used to generate spiking events on specific locations on the cell when the voltage
       increases above a certain threshold; typically act as **sources** of :ref:`connections <modelconnections>`.
     * Gap junction sites: placed on a specific location on a cell and used to electrically couple the cell to
       another gap junction site on another cell by forming a :ref:`gap junction <modelgapjunctions>`.

   Most Arbor users will want to use the cable cell because it is the only cell kind that supports complex
   morphologies and user-defined mechanisms. See the cable cell's :ref:`dedicated page <cablecell>` for more info.
   And visit the :ref:`C++ <cppcablecell>` and :ref:`Python <pycablecell>` APIs to learn how to programmatically
   provide the cable cell description in Arbor.

2. **LIF Cells**

   The description of a LIF cell is used to control the leaky integrate-and-fire dynamics:

      * Resting potential.
      * Reset potential.
      * Initial value of membrane potential.
      * Membrane potential decaying constant.
      * Membrane capacitance.
      * Firing threshold.
      * Refractory period.

   The morphology of a LIF cell is automatically modeled as a single compartment; each cell has one built-in
   **source** and one built-in **target** which do not need to be explicitly added in the cell description.
   LIF cells do not support adding additional **sources** or **targets** to the description. They do not support
   **gap junctions**. They do not support adding density or point mechanisms.

3. **Spiking cells**

   The description of a spiking cell controls the spiking schedule of the cell. Its morphology is
   automatically modeled as a single compartment; each cell has one built-in **source** which does not need to
   be explicitly added in the cell description, and no **targets**. Spiking cells do not support adding additional
   **sources** or **targets**. They do not support **gap junctions**. They do not support adding density or
   point mechanisms.

4. **Benchmark Cells**

   The description of a benchmark cell is used to determine the spiking schedule of the cell and manipulate its
   performance efficiency. This cell is mainly used by developers.

API
---

* :ref:`Python <pycell>`
* :ref:`C++ <cppcell>`
