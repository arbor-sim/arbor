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

    ========================  ================================  ===========================================================
    Identifier                Type                              Description
    ========================  ================================  ===========================================================
    .. generic:: gid          integral                          The global identifier of the cell associated with the item.
    .. generic:: index        integral                          The index of the item in a cell-local collection.
    .. generic:: cell_member  tuple (:gen:`gid`, :gen:`index`)  The global identification of a cell-local item
                                                                associated with a unique cell, identified by the member `gid`,
                                                                and identifying an item within a cell-local collection by the
                                                                member `index`.
    ========================  ================================  ===========================================================


Each cell has a global identifier :gen:`gid` associated to it. The :gen:`gid` is used by the :ref:`recipe <modelrecipe>`
to build the global network. Certain locations on a cell can be pinpointed and potentially used in cell-to-cell interaction.
We identify three kinds of locations on a cell:

1. **Source**
2. **Target**
3. **Gap Junction Site**

A cell has potentially many sources, targets and gap junction sites. Each of these has a local :gen:`index` relative to other points of
the same type on that cell. That cell, as previously mentioned, has a :gen:`gid` to index it relative to other cells in the simulation.
The (:gen:`gid`, :gen:`index`) pairs make up the :gen:`cell_member` type used to index sources, targets and gap junctions sites
globally in a simulation.

Cells may interact with other cells via :ref:`connections <modelconnections>` or
:ref:`gap junctions <modelgapjunctions>`. Connections are formed from **sources** to **targets**. Gap junctions
are formed between two **gap junction sites**.


Cell Kinds
----------

.. table:: Cell Kinds

    ========================  ======================  ===========================================================
    Identifier                Type                    Description
    ========================  ======================  ===========================================================
    cell kind                 enum                    * **cable**: cell with morphology described by branching
                                                        1D cable segments.
                                                      * **lif**: leaky-integrate and fire neuron.
                                                      * **spiking**: spike source from values inserted via
                                                        description.
                                                      * **benchmark**: proxy cell used for benchmarking.
    ========================  ======================  ===========================================================

1. **Cable**

   Cable cells are morphologically detailed cells represented as branching linear 1D segments. They can be coupled
   to other cell types via spike exchange, e.g. a cable cell can receive spikes from a *spiking* cell, and spikes
   form a cable cell can be sent to an *LIF* cell. This coupling happens by two different mechanisms:

   1. Spike exchange over a **connection** with fixed latency.
   2. Direct electrical coupling between two cable cells via **gap junctions**.

   Key concepts:

   * **Morphology**: The morphology of a cable cell is built from the soma outwards; a child branch cannot be constructed before its parent.
     Density mechanisms can be added to already constructed cables.
   * **Detectors**: These refer to the **sources** of :ref:`connections <modelconnections>`.
     They are declared by specifying a location on a branch of the cell, and a threshold voltage for spike detection.
   * **Synapses**: These refer to the **targets** of :ref:`connections <modelconnections>`.
     They are declared by specifying a location on a branch of the cell, and a synapse (point) mechanism.
   * **Gap Junction Sites**: These refer to the sites of :ref:`gap junctions <modelgapjunctions>`.
     They are declared by specifying a location on a branch of the cell.

2. **LIF**

   Leaky integrate and fire neuron. It is a point neuron with one built-in **source** and one built-in **target**.
   It does not support adding additional **sources** or **targets**. It does not support gap junctions.

3. **Spiking**

   Spike source from values inserted via a `schedule description`. It is a point neuron with one built-in **source** and no **targets**.
   It does not support adding additional **sources** or **targets**. It does not support gap junctions.

4. **Benchmark**

   Proxy cell used for benchmarking. Similarly to a spiking cell, a benchmark cell generates spikes according to values
   inserted via a `schedule description`. It also accepts a `realtime ratio` parameter that represents the ratio of
   real cell advancement time to simulation time (if equal to 1, then a single cell can be advanced in realtime).
   A benchmark cell has one built-in **source** and one built-in **target**.
   It does not support adding additional **sources** or **targets**. It does not support gap junctions.

.. _modelconnections:

Connections
===========

Connections represent one of the two types of cell interactions supported in Arbor (the other being :ref:`gap junctions <modelgapjunctions>`).
They implement chemical synapses between **source** and **target** cells and are characterized by having a transmission delay.

Connections in Arbor are defined in two steps:

1. **Source** and **Target** instantiation on the cells: A connection is formed between two locations on two cells.
   These locations need to be declared on the :ref:`cell <modelcells>`.
2. Connection instantiation in the :ref:`recipe <modelrecipe>`: The **sources** and **targets** are indexed using :gen:`cell_member`,
   which identifies a specific instance (:gen:`index`) on a specific cell (:gen:`gid`). A connection is
   instantiated by providing the **source** :gen:`cell_member` and the **target** :gen:`cell_member`, as well as a weight.
   In the recipe, each cell has access to all of the connections whose **targets** are on that cell.

.. _modelgapjunctions:

Gap Junctions
=============

Gap Junctions represent one of the two types of cell interactions supported in Arbor (the other being :ref:`modelconnections`).
They implement electrical synapses where transmission between cells is bidirectional and faster than chemical synapses though with lower gain.
They are modeled as a conductance between two **gap junction sites** on two cells.

Similarly to `Connections`, Gap Junctions in Arbor are defined in two steps:

1. **Gap junction site** instantiation on the cells: A gap junction is formed between two locations on two cells.
   These locations need to be declared on the :ref:`cell <modelcells>`.
2. Gap Junction instantiation in the :ref:`recipe <modelrecipe>`: The **gap junction sites** are indexed using :gen:`cell_member`,
   which identifies a specific instance (:gen:`index`) on a specific cell (:gen:`gid`). A gap junction is
   instantiated by providing 2 **gap junction sites'** :gen:`cell_member`, as well as a conductance in μS.
   In the recipe, each cell has access to all of the gap junctions where at least one :gen:`cell_member::gid` refers to that cell.

.. Note::
   Arbor has Gap Junctions implemented only for cable cells as of now.