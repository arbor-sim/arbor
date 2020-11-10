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
A unique (:gen:`gid`, :gen:`index`) pair defined by a :gen:`cell_member` can be used to uniquely identify
objects on a cell in a global model.


.. _model_cell_kinds:

Cell kinds
----------

.. table:: The 4 types of cell supported by Arbor

    ========================  ===========================================================
    Cell Kind                 Description
    ========================  ===========================================================
    **cable**                 Cell with morphology described by branching
                              1D cable segments and user configurable mechanisms.
    **lif**                   Single-compartment no-mechanism leaky integrate-and-fire
                              neuron.
    **spiking**               Proxy cell that generates spikes from a user-supplied
                              time sequence.
    **benchmark**             Proxy cell used for benchmarking (developer use only).
    ========================  ===========================================================

.. _model_cable_cell:
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
     passes a threshold. Detectors act as **sources** of :ref:`connections <modelconnections>`.
   * **Synapses**: Synapses act as **targets** of :ref:`connections <modelconnections>`.
     A synapse is described by a synapse type (with associated parameters) and location on a cell.
   * **Gap Junction Sites**: These refer to the sites of :ref:`gap junctions <modelgapjunctions>`.
     They are declared by specifying a location on a branch of the cell.

   Because cable cells are the main cell kind in Arbor and have more properties than listed here, they have a
   :ref:`dedicated page <cablecell>`.

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

Most Arbor users will want to use the cable cell, which is the only cell kind that supports complex morphologies
and user-defined mechanisms. You can visit the :ref:`cable cell page <cablecell>` for more information.

API
---

* :ref:`Python <pycell>`
* :ref:`C++ <cppcell>`
