.. _modelcommon:

Common Types
=================

The basic unit of abstraction in an Arbor model is the cell.
A cell represents the smallest model that can be simulated.
Cells interact with each other only via spike exchange.
Cells can be of various types, admitting different representations and implementations.
A *cell group* represents a collection of cells of the same type together with an implementation of their simulation.
Arbor currently supports specialized leaky integrate and fire cells and cells representing artificial spike sources in addition to multi-compartment neurons.

Since the neuron model and the associated workflow are formulated from a cell-centered perspective, cell identifiers and indexes need to be utilized.

.. table:: Cell identifiers and indexes

    ========================  ======================  ===========================================================
    Identifyer/ Index         Type                    Description
    ========================  ======================  ===========================================================
    `gid`                     integer                 The global identifier of the cell associated with the item.
    `index`                   unsigned integer        The index of the item in a cell-local collection.
    cell member               tuple (gid,index)       The global identification of a cell-local item
                                                      associated with a unique cell, identified by the member `gid`,
                                                      and identifying an item within a cell-local collection by the member `index`.
    cell size                 unsigned integer        Counting collections of cells.
    cell local size           unsigned integer        Counting cell-local data.
    cell kind                 enumerator              The identification of the cell type/kind,
                                                      used by the model to group equal kinds in the same cell group:

                                                      - Cell with morphology described by branching 1D cable segments,
                                                      - Leaky-integrate and fire neuron,
                                                      - Regular spiking source,
                                                      - Spike source from values inserted via description.
    ========================  ======================  ===========================================================

Example
    An example of the `cell member` identifyer is uniquely identifying a synapse in the model.
    Each synapse has a post-synaptic cell (`gid`), and an index (`index`) into the set of synapses on the post-synaptic cell.


Further, to interact with the model **probes** are specified whereby the item or value that is subjected to a probe will be specific to a particular cell type.
Probes are specified in the recipe that is used to initialize a model with cell `gid` and index of the probe.
The probe's adress is a cell-type specific location info, specific to the cell kind of `gid`.




C++ specific common types are explained in detail in :ref:`cppcommon` and in :ref:`pycommon` for Arbor's python front end.
