.. _pycell:

Cells
=====================

Cell identifiers and indexes
----------------------------
The types defined below are used as identifiers for cells and members of cell-local collections.

.. module:: arbor

.. class:: cell_member

    .. function:: cell_member(gid, index)

        Construct a ``cell_member`` object with parameters :attr:`gid` and :attr:`index` for global identification of a cell-local item.

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

Cell kinds
----------

.. class:: lif_cell

    A benchmarking cell (leaky integrate-and-fire), used by Arbor developers to test communication performance,
    with neuronal parameters:

    .. attribute:: tau_m

        Membrane potential decaying constant [ms].

    .. attribute:: V_th

        Firing threshold [mV].

    .. attribute:: C_m

        Membrane capacitance [pF].

    .. attribute:: E_L

        Resting potential [mV].

    .. attribute:: V_m

        Initial value of the Membrane potential [mV].

    .. attribute:: t_ref

        Refractory period [ms].

    .. attribute:: V_reset

        Reset potential [mV].

.. class:: spike_source_cell

    A spike source cell, that generates a user-defined sequence of spikes
    that act as inputs for other cells in the network.

    .. function:: spike_source_cell(schedule)

        Construct a spike source cell that generates spikes

        - at regular intervals (using an :class:`arbor.regular_schedule`)
        - at a sequence of user-defined times (using an :class:`arbor.explicit_schedule`)
        - at times defined by a Poisson sequence (using an :class:`arbor.poisson_schedule`)

        :param schedule: User-defined sequence of time points (choose from :class:`arbor.regular_schedule`, :class:`arbor.explicit_schedule`, or :class:`arbor.poisson_schedule`).

.. class:: benchmark_cell

    A benchmarking cell, used by Arbor developers to test communication performance.

    .. function:: benchmark_cell(schedule, realtime_ratio)

        A benchmark cell generates spikes at a user-defined sequence of time points:

        - at regular intervals (using an :class:`arbor.regular_schedule`)
        - at a sequence of user-defined times (using an :class:`arbor.explicit_schedule`)
        - at times defined by a Poisson sequence (using an :class:`arbor.poisson_schedule`)

        and the time taken to integrate a cell can be tuned by setting the parameter ``realtime_ratio``.

        :param schedule: User-defined sequence of time points (choose from :class:`arbor.regular_schedule`, :class:`arbor.explicit_schedule`, or :class:`arbor.poisson_schedule`).

        :param realtime_ratio: Time taken to integrate a cell, for example if ``realtime_ratio`` = 2, a cell will take 2 seconds of CPU time to simulate 1 second.

.. class:: cable_cell
    :noindex:

    See :ref:`pycablecell`.
