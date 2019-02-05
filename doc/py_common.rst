.. _pycommon:

Common Types
=====================

Cell Identifiers and Indexes
----------------------------
These types defined below are used as identifiers for cells and members of cell-local collections.

.. module:: arbor

.. class:: cell_member

    .. function:: cell_member()

        Constructor.

    .. function:: cell_member(gid, index)

        Constructor with parameters :attr:`gid` and :attr:`index`.
        For global identification of an item of a cell-local item.

        Items of type :class:`cell_member` must:

        * be associated with a unique cell, identified by the member :attr:`gid`;
        * identify an item within a cell-local collection by the member :attr:`index`.

        An example is uniquely identifying a synapse in the model.
        Each synapse has a post-synaptic cell (:attr:`gid`), and an index (:attr:`index`) into the set of synapses on the post-synaptic cell.

        Lexographically ordered by :attr:`gid`, then :attr:`index`.

    .. attribute:: gid

        The global identifier of the cell.

    .. attribute:: index

        Cell-local index of the item.
        Local indices for items within a particular cell-local collection should be zero-based and numbered contiguously.

    An example of a cell member construction reads as follows:

    .. container:: example-code

        .. code-block:: python

            import arbor

            # construct
            cmem1 = arbor.cell_member()
            cmem2 = arbor.cell_member(1, 1)

            # set gid and index
            cmem1.gid = 0
            cmem1.index = 0


.. class:: cell_kind

    Enumeration used to indentify the cell type/kind, used by the model to
    group equal kinds in the same cell group.

    .. attribute:: cable1d

        A cell with morphology described by branching 1D cable segments.

    .. attribute:: lif

        Leaky-integrate and fire neuron.

    .. attribute:: spike_source

        Cell that generates spikes at a user-supplied sequence of time points.

    .. attribute:: benchmark

        Proxy cell used for benchmarking.

An example of a cell construction of :class:`cell_kind.cable1d` reads as follows:

    .. container:: example-code

        .. code-block:: python

            import arbor

            kind = arbor.cell_kind.cable1d

Probes
------

Yet to be implemented.
