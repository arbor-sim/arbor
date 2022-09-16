.. _modelcells:

Cells
=====

The basic unit of abstraction in an Arbor model is a cell.
A cell represents the smallest model that can be simulated.
Cells interact with each other via spike exchange and gap junctions.

.. table:: Identifiers used to refer to cells and objects like synapses on cells.

    =============================  ===========================================  ===========================================================
    Identifier                     Type                                         Description
    =============================  ===========================================  ===========================================================
    .. generic:: gid               integral                                     The unique global identifier of a cell.
    .. generic:: tag               string                                       The label of a group of items in a cell-local collection.
                                                                                For example the synapses "syns_0" on a cell.
    .. generic:: selection_policy  enum                                         The policy for selecting a single item out of a group
                                                                                identified by its label.
    .. generic:: local_label       tuple (:gen:`tag`, :gen:`selection_policy`)  The local identification of an cell-local item from a
                                                                                cell-local collection on an unspecified cell.
    .. generic:: global_label      tuple (:gen:`gid`, :gen:`local_lable`)       The global identification of a cell-local item from a
                                                                                cell-local collection on the cell identified by `gid`.
    =============================  ===========================================  ===========================================================

Cell interactions via :ref:`connections <modelconnections>` and :ref:`gap junctions <modelgapjunctions>` occur
between **source**, **target** and **gap junction site** locations on a cell. Connections are formed from sources
to targets. Gap junctions are formed between two gap junction sites. An example of a source on a
:ref:`cable cell<modelcablecell>` is a :ref:`threshold detector <cablecell-threshold-detectors>`;
an example of a target on a cable cell is a :ref:`synapse <cablecell-synapses>`.

**Sources**, **targets** and **gap junction sites** are placed on sets of one or more locations on a cell.
The number of locations in each set (and hence the number of sources/targets/gap junctions), depends on the cell
description. For example, a user may choose to place a synapse at the end of every branch of a cell: the number of
synapses in this case depends on the underlying morphology.

A set of one or more items of the same type (source/target/gap junction) are grouped under a label which can
be when used when forming connections in a network. However, connections are one-to-one, so a :gen:`selection_policy`
is needed to select an item of the group, for both ends of a connection or gap junction.

The combination of :gen:`tag` and :gen:`selection_policy` forms a :gen:`local_label`. When the global identifier of
the cell :gen:`gid` is added, a :gen:`global_label` is formed, capable of globally identifying a source, target or
gap junction site in the network. These :gen:`global_labels` are used to form connections and gap junctions in the
:ref:`recipe <modelrecipe>`.

The :gen:`gid` of a cell is also used to determine its cell :ref:`kind <modelcellkind>` and
:ref:`description <modelcelldesc>` in the :ref:`recipe <modelrecipe>`.

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
