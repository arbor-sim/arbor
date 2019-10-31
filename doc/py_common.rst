.. _pycommon:

Common Types
=====================

Cell Identifiers and Indexes
----------------------------
The types defined below are used as identifiers for cells and members of cell-local collections.

.. module:: arbor

.. class:: cell_member

    .. function:: cell_member(gid, index)

        Construct a cell member with parameters :attr:`gid` and :attr:`index` for global identification of a cell-local item.

        Items of type :class:`cell_member` must:

        * be associated with a unique cell, identified by the member :attr:`gid`;
        * identify an item within a cell-local collection by the member :attr:`index`.

        An example is uniquely identifying a synapse in the model.
        Each synapse has a post-synaptic cell (with :attr:`gid`), and an :attr:`index` into the set of synapses on the post-synaptic cell.

        Lexographically ordered by :attr:`gid`, then :attr:`index`.

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

Morphology
----------

.. class:: location

    Construct a location specification with the :attr:`branch` id and the relative :attr:`position` on the branch ∈ [0.,1.], where 0. means proximal and 1. distal.

    .. attribute:: branch

        The id of the branch.

    .. attribute:: position

        The relative position (from 0., proximal, to 1., distal) on the branch.

