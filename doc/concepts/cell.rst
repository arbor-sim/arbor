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

   LIF cells are :term:`single-compartment <control volume>` leaky integrate and fire neurons. They are typically used to simulate
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

   See :ref:`cablecell`.

2. **LIF Cells**

   See :ref:`lifcell`.

3. **Spiking cells**

   See :ref:`spikecell`.

4. **Benchmark Cells**

   See :ref:`benchcell`.

API
---

* :ref:`Python <pycell>`
* :ref:`C++ <cppcell>`
