.. _modelcommon:

Common Types
=================

The basic unit of abstraction in an Arbor model is a cell.
A cell represents the smallest model that can be simulated.
Cells interact with each other via spike exchange and gap junctions.
Cells can be of various types, admitting different representations and implementations.
A *cell group* represents a collection of cells of the same type together with an implementation of their simulation.
Arbor currently supports specialized leaky integrate and fire cells and cells representing artificial spike sources in addition to multi-compartment neurons.

Since the neuron model and the associated workflow are formulated from a cell-centered perspective, cell identifiers and indexes need to be utilized.

.. table:: Cell identifiers and indexes

    ========================  ======================  ===========================================================
    Identifyer/ Index         Type                    Description
    ========================  ======================  ===========================================================
    gid                       integral                The global identifier of the cell associated with the item.
    index                     integral                The index of the item in a cell-local collection.
    cell member               tuple (gid, index)      The global identification of a cell-local item
                                                      associated with a unique cell, identified by the member `gid`,
                                                      and identifying an item within a cell-local collection by the member `index`.
    cell size                 integral                Counting collections of cells.
    cell local size           integral                Counting cell-local data.
    cell kind                 eunum                   Identifiies of the cell type, including:

                                                      * Cell with morphology described by branching 1D cable segments.
                                                      * Leaky-integrate and fire neuron.
                                                      * Regular spiking source.
                                                      * Spike source from values inserted via description.
    ========================  ======================  ===========================================================

Example
    A `cell member` identifier is used to uniquely identify synapses.
    Each synapse has a post-synaptic cell (`gid`), and an `index` into the set of synapses on the post-synaptic cell.

C++ specific common types are explained in detail in :ref:`cppcommon`.
