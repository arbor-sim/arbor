.. _cppcommon:

Common Types
============

.. cpp:namespace:: arb

Cell Identifiers and Indexes
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

Probes
------

.. cpp:type:: probe_tag = int

    Extra contextual information associated with a probe.

.. cpp:class:: probe_info

    Probes are specified in the recipe objects that are used to initialize a
    model; the specification of the item or value that is subjected to a
    probe will be specific to a particular cell type.

    .. cpp:member:: cell_member_type id

           Cell gid, index of probe.

    .. cpp:member:: probe_tag tag

           Opaque key, returned in sample record.

    .. cpp:member:: util::any address

           Cell-type specific location info, specific to cell kind of ``id.gid``.

Utility Wrappers and Containers
--------------------------------

.. cpp:namespace:: arb::util


.. cpp:class:: template <typename T> optional

    A wrapper around a contained value of type :cpp:type:`T`, that may or may not be set.
    A faithful copy of the C++17 ``std::optional`` type.
    See the online C++ standard documentation
    `<https://en.cppreference.com/w/cpp/utility/optional>`_
    for more information.

.. cpp:class:: any

    A container for a single value of any type that is copy constructable.
    Used in the Arbor API where a type of a value passed to or from the API
    is decided at run time.

    A faithful copy of the C++17 ``std::any`` type.
    See the online C++ standard documentation
    `<https://en.cppreference.com/w/cpp/utility/any>`_
    for more information.

    The :cpp:any:`arb::util` namespace also implementations of the
    :cpp:any:`any_cast`, :cpp:any:`make_any` and :cpp:any:`bad_any_cast`
    helper functions and types from C++17.

.. cpp:class:: unique_any

   Equivalent to :cpp:class:`util::any`, except that:
   
      * it can store any type that is move constructable;
      * it is move only, that is it can't be copied.


