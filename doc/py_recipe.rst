.. _pyrecipe:

Recipes
=================

.. currentmodule:: arbor

The :class:`recipe` class documentation is below.

A recipe describes neuron models in a cell-oriented manner and supplies methods to provide cell information.
Details on why Arbor uses recipes and general best practices can be found in :ref:`modelrecipe`.

.. class:: recipe

    Describe a model by describing the cells and network, without any information about how the model is to be represented or executed.

    All recipes derive from this abstract base class.

    Recipes provide a cell-centric interface for describing a model.
    This means that model properties, such as connections, are queried using the global identifier (:attr:`arbor.cell_member.gid`) of a cell.
    In the description below, the term ``gid`` is used as shorthand for the cell with global identifier.

    **Required Member Functions**

    The following member functions (besides a constructor) must be implemented by every recipe:

    .. function:: num_cells()

        The number of cells in the model.

    .. function:: cell_kind(gid)

        The cell kind of the cell with global identifier :attr:`arbor.cell_member.gid` (return type: :class:`arbor.cell_kind`).

    .. function:: cell_description(gid)

        A high level decription of the cell with global identifier :attr:`arbor.cell_member.gid`,
        for example the morphology, synapses and ion channels required to build a multi-compartment neuron.
        The type used to describe a cell depends on the kind of the cell.
        The interface for querying the kind and description of a cell are seperate
        to allow the cell type to be provided without building a full cell description,
        which can be very expensive.

    **Optional Member Functions**

    .. function:: connections_on(gid)

        Returns a list of all the **incoming** connections to :attr:`arbor.cell_member.gid`.
        Each connection should have post-synaptic target ``connection.dest.gid``
        that matches the argument :attr:`arbor.cell_member.gid`,
        and a valid synapse id ``connection.dest.index`` on :attr:`arbor.cell_member.gid`.
        See :class:`connection`.

        By default returns an empty list.

    .. function:: gap_junctions_on(gid)

        Returns a list of all the gap junctions connected to :attr:`arbor.cell_member.gid`.
        Each gap junction ``gj`` should have one of the two gap junction sites ``gj.local.gid``
        or ``gj.peer.gid`` matching the argument :attr:`arbor.cell_member.gid`,
        and the corresponding synapse id ``gj.local.index`` or ``gj.peer.index`` should be valid on :attr:`arbor.cell_member.gid`.
        See :class:`gap_junction_connection`.

        By default returns an empty list.

    .. function:: event_generators(gid)

        A list of all the :class:`event_generator` s that are attached to :attr:`arbor.cell_member.gid`.

        By default returns an empty list.

    .. function:: num_sources(gid)

        The number of spike sources on :attr:`arbor.cell_member.gid`.

        By default returns 0.

    .. function:: num_targets(gid)

        The number of post-synaptic sites on :attr:`arbor.cell_member.gid`, which corresponds to the number of synapses.

        By default returns 0.

    .. function:: num_probes(gid)

        The number of probes attached to the cell with :attr:`arbor.cell_member.gid`.

        By default returns 0.

    .. function:: num_gap_junction_sites(gid)

        Returns the number of gap junction sites on :attr:`arbor.cell_member.gid`.

        By default returns 0.

    .. function:: get_probe(id)

        Returns the probe(s) to allow monitoring.

        By default throws a runtime error. If :func:`num_probes`
        returns a non-zero value, this must also be overridden.

.. class:: probe

        Describes the cell probe's information.

.. function:: cable_probe(kind, id, location)

        Returns the description of a probe at an :class:`arbor.location` on a cable cell with :attr:`id` available for monitoring data of ``voltage`` or ``current`` :attr:`kind`.

        An example of a probe on a cable cell for measuring voltage at the soma reads as follows:

    .. container:: example-code

        .. code-block:: python

            import arbor

            id    = arbor.cell_member(0, 0) # cell 0, probe 0
            loc   = arbor.location(0, 0)    # at the soma
            probe = arbor.cable_probe('voltage', id, loc)

.. class:: connection

    Describes a connection between two cells:
    Defined by source and destination end points (that is pre-synaptic and post-synaptic respectively),
    a connection weight and a delay time.

    .. function:: connection(source, destination, weight, delay)

        Construct a connection between the :attr:`source` and the :attr:`dest` with a :attr:`weight` and :attr:`delay`.

    .. attribute:: source

        The source end point of the connection (type: :class:`arbor.cell_member`).

    .. attribute:: dest

        The destination end point of the connection (type: :class:`arbor.cell_member`).

    .. attribute:: weight

        The weight delivered to the target synapse.
        The weight is dimensionless, and its interpretation is specific to the type of the synapse target.
        For example, the expsyn synapse interprets it as a conductance with units μS (micro-Siemens).

    .. attribute:: delay

        The delay time of the connection [ms]. Must be positive.

    An example of a connection reads as follows:

    .. container:: example-code

        .. code-block:: python

            import arbor

            # construct a connection between cells (0,0) and (1,0) with weight 0.01 and delay of 10 ms.
            src  = arbor.cell_member(0,0)
            dest = arbor.cell_member(1,0)
            w    = 0.01
            d    = 10
            con  = arbor.connection(src, dest, w, d)

.. class:: gap_junction_connection

    Describes a gap junction between two gap junction sites.
    Gap junction sites are represented by :class:`arbor.cell_member`.

    .. function::gap_junction_connection(local, peer, ggap)

        Construct a gap junction connection between :attr:`local` and :attr:`peer` with conductance :attr:`ggap`.

    .. attribute:: local

        The gap junction site: one half of the gap junction connection.

    .. attribute:: peer

        The gap junction site: other half of the gap junction connection.

    .. attribute:: ggap

        The gap junction conductance [μS].

Event Generator and Schedules
-----------------------------

.. class:: event_generator

    .. function:: event_generator(target, weight, schedule)

        Construct an event generator for a :attr:`target` synapse with :attr:`weight` of the events to deliver based on a schedule (i.e., :class:`arbor.regular_schedule`, :class:`arbor.explicit_schedule`, :class:`arbor.poisson_schedule`).

    .. attribute:: target

        The target synapse of type :class:`arbor.cell_member`.

    .. attribute:: weight

        The weight of events to deliver.

.. class:: regular_schedule

    Describes a regular schedule with multiples of :attr:`dt` within the interval [:attr:`tstart`, :attr:`tstop`).

    .. function:: regular_schedule(tstart, dt, tstop)

        Construct a regular schedule as list of times from :attr:`tstart` to :attr:`tstop` in :attr:`dt` time steps.

        By default returns a schedule with :attr:`tstart` = :attr:`tstop` = ``None`` and :attr:`dt` = 0 ms.

    .. attribute:: tstart

        The delivery time of the first event in the sequence [ms].
        Must be non-negative or ``None``.

    .. attribute:: dt

        The interval between time points [ms].
        Must be non-negative.

    .. attribute:: tstop

        No events delivered after this time [ms].
        Must be non-negative or ``None``.

    .. function:: events(t0, t1)

        Returns a view of monotonically increasing time values in the half-open interval [t0, t1).

.. class:: explicit_schedule

    Describes an explicit schedule at a predetermined (sorted) sequence of :attr:`times`.

    .. function:: explicit_schedule(times)

        Construct an explicit schedule.

        By default returns a schedule with an empty list of times.

    .. attribute:: times

        The list of non-negative times [ms].

    .. function:: events(t0, t1)

        Returns a view of monotonically increasing time values in the half-open interval [t0, t1).

.. class:: poisson_schedule

    Describes a schedule according to a Poisson process.

    .. function:: poisson_schedule(tstart, freq, seed)

        Construct a Poisson schedule.

        By default returns a schedule with events starting from :attr:`tstart` = 0 ms,
        with an expected frequency :attr:`freq` = 10 Hz and :attr:`seed` = 0.

    .. attribute:: tstart

        The delivery time of the first event in the sequence [ms].

    .. attribute:: freq

        The expected frequency [Hz].

    .. attribute:: seed

        The seed for the random number generator.

    .. function:: events(t0, t1)

        Returns a view of monotonically increasing time values in the half-open interval [t0, t1).

An example of an event generator reads as follows:

.. container:: example-code

    .. code-block:: python

        import arbor

        # define a Poisson schedule with start time 1 ms, expected frequency of 5 Hz,
        # and the target cell's gid as seed
        target = arbor.cell_member(0,0)
        seed   = target.gid
        tstart = 1
        freq   = 5
        sched  = arbor.poisson_schedule(tstart, freq, seed)

        # construct an event generator with this schedule on target cell and weight 0.1
        w      = 0.1
        gen    = arbor.event_generator(target, w, sched)

Cells
------

.. class:: cable_cell

   See :ref:`pycable_cell`.

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

Below is an example of a recipe construction of a ring network of multi-compartmental cells.
Because the interface for specifying cable morphology cells is under construction, the temporary
helpers in cell_parameters and make_cable_cell for building cells are used.

.. container:: example-code

    .. code-block:: python

        import sys
        import arbor

        class ring_recipe (arbor.recipe):

            def __init__(self, n=4):
                # The base C++ class constructor must be called first, to ensure that
                # all memory in the C++ class is initialized correctly.
                arbor.recipe.__init__(self)
                self.ncells = n
                self.params = arbor.cell_parameters()

            # The num_cells method that returns the total number of cells in the model
            # must be implemented.
            def num_cells(self):
                return self.ncells

            # The cell_description method returns a cell
            def cell_description(self, gid):
                return arbor.make_cable_cell(gid, self.params)

            def num_targets(self, gid):
                return 1

            def num_sources(self, gid):
                return 1

            # The kind method returns the type of cell with gid.
            # Note: this must agree with the type returned by cell_description.
            def cell_kind(self, gid):
                return arbor.cell_kind.cable

            # Make a ring network
            def connections_on(self, gid):
                src = (gid-1)%self.ncells
                w = 0.01
                d = 10
                return [arbor.connection(arbor.cell_member(src,0), arbor.cell_member(gid,0), w, d)]

            # Attach a generator to the first cell in the ring.
            def event_generators(self, gid):
                if gid==0:
                    sched = arbor.explicit_schedule([1])
                    return [arbor.event_generator(arbor.cell_member(0,0), 0.1, sched)]
                return []

            # Define one probe (for measuring voltage at the soma) on the cell.
            def num_probes(self, gid):
                return 1

            def get_probe(self, id):
                loc = arbor.location(0, 0) # at the soma
                return arbor.cable_probe('voltage', id, loc)
