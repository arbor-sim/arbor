.. _modelconcepts:

Concepts
########

This section describes some of the core concepts of Arbor including cell definitions and interactions.

.. _modelcells:

Cells
=====

The basic unit of abstraction in an Arbor model is a cell.
A cell represents the smallest model that can be simulated.
Cells interact with each other via spike exchange and gap junctions.
Cells can be of various types, admitting different representations and implementations.
A *cell group* represents a collection of cells of the same type together with an implementation of their simulation.
Arbor currently supports specialized leaky integrate and fire cells and cells representing artificial spike sources in
addition to multi-compartment neurons.

Common Types
------------

.. table:: Cell identifiers

    ========================  ======================  ===========================================================
    Identifier                Type                    Description
    ========================  ======================  ===========================================================
    gid                       integral                The global identifier of the cell associated with the item.
    index                     integral                The index of the item in a cell-local collection.
    cell_member               tuple (gid, index)      The global identification of a cell-local item
                                                      associated with a unique cell, identified by the member `gid`,
                                                      and identifying an item within a cell-local collection by the
                                                      member `index`.
    ========================  ======================  ===========================================================


Each cell has a global identifier :cpp:type:`gid` associated to it. The :cpp:type:`gid` is used by the :ref:`recipe <modelrecipe>`
to build the global network. Cells may interact with other cells via :ref:`connections <modelconnections>` or
:ref:`gap junctions <modelgapjunctions>`. These interactions happen at specific positions on the cell which can
be one of three types:

1. **Source**: A spike source associated to a *connection*. It detects spikes; if part of a connection, it sends the spikes to **targets**.
2. **Target**: A synapse associated to a *connection*. It has an associated synaptic mechanism that can respond to spikes from **sources**.
3. **Gap Junction Site**: A site associated to a *gap junction*. It represents one half of the gap junction.

Each of these positions is located on a specific point on a cell. It has an :cpp:type:`index` relative to all other positions of
the same type on that cell. That cell, as previously mentioned, has a :cpp:type:`gid` to index it relative to other cells in the simulation.
The (:cpp:type:`gid`, :cpp:type:`index`) pairs make up the :cpp:type:`cell_member` type used to index **sources**, **targets** and **gap junctions sites**
for all cells in a simulation.


Cell Kinds
----------

.. table:: Cell Kinds

    ========================  ======================  ===========================================================
    Identifier                Type                    Description
    ========================  ======================  ===========================================================
    cell kind                 enum                    * **cable_cell**: cell with morphology described by branching
                                                        1D cable segments.
                                                      * **lif_cell**: leaky-integrate and fire neuron.
                                                      * **spike_source**: spike source from values inserted via
                                                        description.
                                                      * **benchmark**: proxy cell used for benchmarking.
    ========================  ======================  ===========================================================

1. **Cable Cell**

   Cable cells are morphologically detailed cells represented as branching linear 1D segments. They allow the addition of
   point and density mechanisms. They allow having *gap junctions* and *connections* to other cable cells.

   * **Morphology**: The morphology of a cable cell is built from the soma outwards; a child branch cannot be constructed before its parent.
   * **Detectors**: These refer to the **sources** of :ref:`connections <modelconnections>`.
     They are declared by specifying a location on a branch of the cell, and a threshold voltage for spike detection.
     They have a local index (:cpp:type:`cell_member::index`) relative to other detectors on the cell.
   * **Synapses**: These refer to the **targets** of :ref:`connections <modelconnections>`.
     They are declared by specifying a location on a branch of the cell, and a synapse mechanism.
     They have a local index (:cpp:type:`cell_member::index`) relative to other synapses on the cell.
   * **Gap Junction Sites**: These refer to the sites of :ref:`gap junctions <modelgapjunctions>`.
     They are declared by specifying a location on a branch of the cell.
     They have a local index (:cpp:type:`cell_member::index`) relative to other **gap junction sites** on the cell.

2. **LIF Cell**

   Leaky integrate and fire neuron.

3. **Spike Source**

4. **Benchmark**

.. _modelconnections:

Connections
===========

Connections represent one of the two types of cell interactions supported in Arbor (the other being :ref:`gap junctions <modelgapjunctions>`).
They implement chemical synapses between **source** and **target** cells and are typically characterized by having a transmission delay.

Connections in Arbor are defined in two steps:

1. **Source** and **Target** instantiation on the cells: A connection is formed between two locations on two cells.
   These locations need to be declared at the level of the cell.
2. Connection instantiation in the :ref:`recipe <modelrecipe>`: The **sources** and **targets** are indexed using :cpp:type:`cell_member`,
   which identifies a specific instance (:cpp:type:`cell_member::index`) on a specific cell (:cpp:type:`cell_member::gid`). A connection is
   instantiated by providing the **source** :cpp:type:`cell_member` and the **target** :cpp:type:`cell_member`, as well as a weight.
   In the recipe, each cell has access to all of the connections whose **targets** are on that cell.

.. _modelgapjunctions:

Gap Junctions
=============

Gap Junctions represent one of the two types of cell interactions supported in Arbor (the other being :ref:`modelconnections`).
They implement electrical synapses where transmission between cells is bidirectional and faster than chemical synapses though with lower gain.
They are modeled as a conductance between two **gap junction sites** on two cells.

Similarly to `Connections`, Gap Junctions in Arbor are defined in two steps:

1. **Gap junction site** instantiation on the cells: A gap junction is formed between two locations on two cells.
   These locations need to be declared at the level of the cell.
2. Gap Junction instantiation in the :ref:`recipe <modelrecipe>`: The **gap junction sites** are indexed using :cpp:type:`cell_member`,
   which identifies a specific instance (:cpp:type:`cell_member::index`) on a specific cell (:cpp:type:`cell_member::gid`). A gap junction is
   instantiated by providing 2 **gap junction sites'** *cell_members*, as well as a conductance in μS.
   In the recipe, each cell has access to all of the gap junctions where at least one :cpp:type:`cell_member::gid` refers to that cell.

Arbor has Gap Junctions implemented only for cable_cells as of now.