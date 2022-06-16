.. _pycell:

Cells
=====================

Cell identifiers and indexes
----------------------------
The types defined below are used as identifiers for cells and members of cell-local collections.

.. module:: arbor

.. class:: selection_policy

   Enumeration used for selecting an individual item from a group of items sharing the
   same label.

   .. attribute:: round_robin

      Iterate over the items of the group in a round-robin fashion.

   .. attribute:: round_robin_halt

      Halts at the current item of the group until the round_robin policy is called (again).

   .. attribute:: univalent

      Assert that only one item is available in the group. Throws an exception if the assertion
      fails.

.. class:: cell_local_label

   For local identification of an item on an unspecified cell.

   A local string label :attr:`tag` is used to identify a group of items within a particular
   cell-local collection. Each label is associated with a set of items distributed over various
   locations on the cell. The exact number of items associated to a label can only be known when the
   model is built and is therefore not directly available to the user.

   Because each label can be mapped to any of the items in its group, a :attr:`selection_policy`
   is needed to select one of the items of the group. If the policy is not supplied, the default
   :attr:`selection_policy.univalent` is selected.

   :class:`cell_local_label` is used for selecting the target of a connection or the
   local site of a gap junction connection. The cell ``gid`` of the item is implicitly known in the
   recipe.

   .. attribute:: tag

      Identifier of a group of items in a cell-local collection.

   .. attribute:: selection_policy

      Policy used for selecting a single item of the tagged group.

   An example of a cell member construction reads as follows:

   .. container:: example-code

       .. code-block:: python

           import arbor

           # Create the policy
           policy = arbor.selection_policy.univalent

           # Create the local label referring to the group of items labeled "syn0".
           # The group is expected to only contain 1 item.
           local_label = arbor.cell_local_label("syn0", policy)

.. class:: cell_global_label

   For global identification of an item on a cell.
   This is used for selecting the source of a connection or the peer site of a gap junction connection.
   The :attr:`label` expects a :class:`cell_local_label` type.

   .. attribute:: gid

      Global identifier of the cell associated with the item.

   .. attribute:: label

      Identifier of a single item on the cell.

   .. container:: example-code

       .. code-block:: python

           import arbor

           # Create the policy
           policy = arbor.selection_policy.univalent

           # Creat the local label referring to the group of items labeled "syn0".
           # The group is expected to only contain 1 item.
           local_label = arbor.cell_local_label("syn0", policy)

           # Create the global label referring to the group of items labeled "syn0"
           # on cell 5
           global_label = arbor.cell_global_label(5, local_label)

.. class:: cell_member

    .. function:: cell_member(gid, index)

        Construct a ``cell_member`` object with parameters :attr:`gid` and :attr:`index` for
        global identification of a cell-local item.

        Items of type :class:`cell_member` must:

        * be associated with a unique cell, identified by the member :attr:`gid`;
        * identify an item within a cell-local collection by the member :attr:`index`.

        An example is uniquely identifying a probeset on the model:
        ``arbor.cell_member(12, 3)`` can be used to identify the probeset with :attr:`index` 3 on the cell with :attr:`gid` 12.

        Lexicographically ordered by :attr:`gid`, then :attr:`index`.

    .. attribute:: gid

        The global identifier of the cell.

    .. attribute:: index

        The cell-local index of the item.
        Local indices for items within a particular cell-local collection should be zero-based and numbered contiguously.

    An example of a cell member construction reads as follows:

    .. container:: example-code

        .. code-block:: python

            import arbor

            # construct
            cmem = arbor.cell_member(0, 0)

            # set gid and index
            cmem.gid = 1
            cmem.index = 42

.. class:: cell_kind

    Enumeration used to identify the cell kind, used by the model to group equal kinds in the same cell group.

    .. attribute:: cable

        A cell with morphology described by branching 1D cable segments.

    .. attribute:: lif

        A leaky-integrate and fire neuron.

    .. attribute:: spike_source

        A proxy cell that generates spikes from a spike sequence provided by the user.

    .. attribute:: benchmark

        A proxy cell used for benchmarking.

    An example for setting the cell kind reads as follows:

    .. container:: example-code

        .. code-block:: python

            import arbor

            kind = arbor.cell_kind.cable

Cell kinds
----------

.. class:: lif_cell
    :noindex:

    See :ref:`pylifcell`.

.. class:: spike_source_cell
    :noindex:

    See :ref:`pyspikecell`.

.. class:: benchmark_cell
    :noindex:

    See :ref:`pybenchcell`.

.. class:: cable_cell
    :noindex:

    See :ref:`pycablecell`.
