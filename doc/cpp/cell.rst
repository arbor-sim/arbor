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
    :cpp:type:`cell_local_size_type`; and uses ``std::string`` for
    :cpp:type:`cell_tag_type` at the time of writing. However,
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

.. cpp:type::  cell_tag_type

    For labels of cell-local data.
    Local labels are used for groups of items within a particular cell-local collection.
    Each label is associated with a range of :cpp:type:`cell_lid_type` indexing the individual
    items on the cell. The range is generated when the model is built and is not directly
    available to the user.

.. cpp:enum::  lid_selection_policy

   For selecting an individual item from a group of items sharing the
   same :cpp:type:`cell_tag_type` label.

   .. cpp:enumerator:: round_robin

      Iterate over the items of the group in a round-robin fashion.

   .. cpp:enumerator:: round_robin_halt

      Halts at the current item of the group until the round_robin policy is called (again).

   .. cpp:enumerator:: assert_univalent

      Assert that ony one item is available in the group. Throws an exception if the assertion
      fails.

.. cpp:class::  cell_local_label_type

   For local identification of an item on an unspecified cell.
   This is used for selecting the target of a connection or the local site of a gap junction
   connection. The cell ``gid`` is implicitly known from the recipe.

   .. cpp:member:: cell_tag_type  tag

      Identifier of a group of items in a cell-local collection.

   .. cpp:member:: lid_selection_policy   policy

      Policy used for selecting a single item of the tagged group.

.. cpp:class::  cell_global_label_type

   For global identification of an item on a cell.
   This is used for selecting the source of a connection or the peer site of a gap junction
   connection.

   .. cpp:member:: cell_gid_type   gid

      Global identifier of the cell associated with the item.

   .. cpp:member:: cell_local_label_type label

      Identifier of a single item on the cell.

.. cpp:type::  cell_local_size_type

    An unsigned integer for for counts of cell-local data.

.. cpp:class:: cell_member_type

    For global identification of an item of cell local data.
    Items of :cpp:type:`cell_member_type` must:

        * be associated with a unique cell, identified by the member
          :cpp:member:`gid`;
        * identify an item within a cell-local collection by the member
          :cpp:member:`index`.

    An example is uniquely identifying a probe description in the model.
    Each probe has a cell id (:cpp:member:`gid`), and an index
    (:cpp:member:`index`) into the set of probes on the cell.

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

