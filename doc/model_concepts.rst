.. _modelconcepts:

Concepts
########

This section describes some of the core concepts of Arbor.

.. _modelcells:

Cells
=====

The basic unit of abstraction in an Arbor model is a cell.
A cell represents the smallest model that can be simulated.
Cells interact with each other via spike exchange and gap junctions.
Cells can be of various types, admitting different representations and implementations.
Arbor currently supports specialized leaky integrate and fire cells and cells representing artificial spike sources in
addition to multi-compartment neurons.

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


Each cell has a global identifier :gen:`gid`, which is used to refer to cells in :ref:`recipes <modelrecipe>`.
To describe or refer to cell-to-cell interactions, the following object types need to be enumerated:

1. **Sources**
2. **Targets**
3. **Gap Junction Sites**

Cells interact with other cells via :ref:`connections <modelconnections>` or
:ref:`gap junctions <modelgapjunctions>`. Connections are formed from **sources** to **targets**. Gap junctions
are formed between two **gap junction sites**.

A cell can have multiple sources, targets and gap junction site objects. Each object has a local :gen:`index`
relative to other objects of the same type on that cell.
A unique (:gen:`gid`, :gen:`index`) pair defned by a :gen:`cell_member` can be used to uniquely identify
objects on a cell in a global model.


Cell Kinds
----------

.. table:: The types of cell supported by Arbor

    ========================  ===========================================================
    Cell Kind                 Description
    ========================  ===========================================================
    **cable**                 Cell with morphology described by branching
                              1D cable segments.
    **lif**                   Leaky-integrate and fire neuron.
    **spiking**               Proxy cell that generates spikes from a user-supplied
                              time sequence.
    **benchmark**             Proxy cell used for benchmarking (developer use only).
    ========================  ===========================================================

1. **Cable Cells**

   Cable cells are morphologically-detailed cells represented as branching linear 1D segments. They can be coupled
   to other cell types via the following mechanisms:

   1. Spike exchange over a **connection** with fixed latency.
      Cable cells can *receive* spikes from any kind of cell, and can be a *source* of spikes
      cells that have target sites (i.e. *cable* and *lif* cells).
   2. Direct electrical coupling between two cable cells via **gap junctions**.

   Key concepts:

   * **Morphology**: The morphology of a cable cell is composed of a branching tree of one-dimensional line segments.
     Strictly speaking, Arbor represents a morphology is an *acyclic directed graph*, with the soma at the root.
   * **Detectors**: Spike detectors generate spikes when the voltage at location on the cell
     passes a threshold. Dectectors act as **sources** of :ref:`connections <modelconnections>`.
   * **Synapses**: Synapases act as **targets** of :ref:`connections <modelconnections>`.
     A synapse is described by a synapse type (with associated parameters) and location on a cell.
   * **Gap Junction Sites**: These refer to the sites of :ref:`gap junctions <modelgapjunctions>`.
     They are declared by specifying a location on a branch of the cell.

2. **LIF Cells**

   A single compartment leaky integrate and fire neuron with one **source** and one **target**.
   LIF cells does not support adding additional **sources** or **targets** or gap junctions.

3. **Spiking Cells**

   Spike source from values inserted via a `schedule description`. It is a point neuron with one built-in **source** and no **targets**.
   It does not support adding additional **sources** or **targets**. It does not support gap junctions.

4. **Benchmark Cells**

   Proxy cell used for benchmarking, and used by developers to benchmark the spike exchange and event delivery infrastructure.

.. _modelconnections:

Connections
===========

Connections implement chemical synapses between **source** and **target** cells and are characterized by having a transmission delay.

Connections in Arbor are defined in two steps:

1. Create **Source** and **Target** on two cells: a source defined on one cell, and a target defined on another.
2. Declare the connection in the :ref:`recipe <modelrecipe>`: with a source and target idenfied using :gen:`cell_member`, a connection delay and a connection weight.

.. _modelgapjunctions:

Gap Junctions
=============

Gap junctions represent electrical synapses where transmission between cells is bidirectional and direct.
They are modeled as a conductance between two **gap junction sites** on two cells.

Similarly to `Connections`, Gap Junctions in Arbor are defined in two steps:

1. A **gap junction site** is created on each of the two cells.
   These locations need to be declared on the :ref:`cell <modelcells>`.
2. Gap Junction instantiation in the :ref:`recipe <modelrecipe>`: The **gap junction sites** are indexed using :gen:`cell_member`
   because a single cell may have more than one gap junction site.
   A gap junction is instantiated by providing two **gap junction sites'** and a conductance in μS.

   .. Note::
      Only cable cells support gap junctions as of now.
