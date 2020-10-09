.. _cppcell:

Cells
============

.. cpp:namespace:: arb

Cell identifiers and indexes
----------------------------

These types, defined in ``common_types.hpp``, are used as identifiers for
cells and members of cell-local collections.

.. Note::
    Arbor uses ``std::unit32_t`` for :cpp:type:`cell_gid_type`,
    :cpp:type:`cell_size_type`, :cpp:type:`cell_lid_type`, and
    :cpp:type:`cell_local_size_type` at the time of writing, however
    this could change, e.g. to handle models that cell gid that don't
    fit into a 32 bit unsigned integer.
    It is thus recommended that these type aliases be used whenever identifying
    or counting cells and cell members.


.. cpp:type:: cell_gid_type

    An integer type used for identifying cells globally.


.. cpp:type::  cell_size_type

    An unsigned integer for sizes of collections of cells.
    Unsigned type for counting :cpp:type:`cell_gid_type`.


.. cpp:type::  cell_lid_type

    For indexes into cell-local data.
    Local indices for items within a particular cell-local collection should be
    zero-based and numbered contiguously.


.. cpp:type::  cell_local_size_type

    An unsigned integer for for counts of cell-local data.


.. cpp:class:: cell_member_type

    For global identification of an item of cell local data.
    Items of :cpp:type:`cell_member_type` must:

        * be associated with a unique cell, identified by the member
          :cpp:member:`gid`;
        * identify an item within a cell-local collection by the member
          :cpp:member:`index`.

    An example is uniquely identifying a synapse in the model.
    Each synapse has a post-synaptic cell (:cpp:member:`gid`), and an index
    (:cpp:member:`index`) into the set of synapses on the post-synaptic cell.

    Lexicographically ordered by :cpp:member:`gid`,
    then :cpp:member:`index`.

    .. cpp:member:: cell_gid_type   gid

        Global identifier of the cell containing/associated with the item.

    .. cpp:member:: cell_lid_type   index

        The index of the item in a cell-local collection.


.. cpp:enum-class:: cell_kind

    Enumeration used to identify the cell type/kind, used by the model to
    group equal kinds in the same cell group.

    .. cpp:enumerator:: cable

        A cell with morphology described by branching 1D cable segments.

    .. cpp:enumerator:: lif

        Leaky-integrate and fire neuron.

    .. cpp:enumerator:: spike_source

        Proxy cell that generates spikes from a spike sequence provided by the user.

    .. cpp:enumerator:: benchmark

        Proxy cell used for benchmarking.

