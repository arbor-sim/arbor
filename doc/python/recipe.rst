.. _pyrecipe:

Recipes
=================

.. currentmodule:: arbor

The :class:`recipe` class documentation is below.

A recipe describes neuron models in a cell-oriented manner and supplies methods to provide cell information.
Details on why Arbor uses recipes and general best practices can be found in :ref:`modelrecipe`.

Recipe
------

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

        A high level description of the cell with global identifier :attr:`arbor.cell_member.gid`,
        for example the morphology, synapses and ion channels required to build a multi-compartment neuron.
        The type used to describe a cell depends on the kind of the cell.
        The interface for querying the kind and description of a cell are separate
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

    .. function:: num_gap_junction_sites(gid)

        Returns the number of gap junction sites on :attr:`arbor.cell_member.gid`.

        By default returns 0.

    .. function:: probes(gid)

        Returns a list specifying the probe addresses describing probes on the cell ``gid``.
        Each address in the list is an opaque object of type :class:`probe` produced by
        cell kind-specific probe address functions. Each probe address in the list
        has a corresponding probe id of type :class:`cell_member_type`: an id ``(gid, i)``
        refers to the probes described by the ith entry in the list returned by ``get_probes(gid)``.

        By default returns an empty list.

    .. function:: global_properties(kind)

        The global properties of a model.

        This method needs to be implemented for :class:`arbor.cell_kind.cable`, where the
        properties include ion concentrations and reversal potentials; initial membrane voltage;
        temperature; axial resistivity; membrane capacitance; cv_policy; and a pointer
        to the mechanism catalogue.

        By default returns an empty object.

Cells
------

See :ref:`pycell`.

Synapses
--------

See :ref:`pyinterconnectivity`.

Event generator and schedules
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

Example
-------

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

            # The cell_description method returns a cell.
            def cell_description(self, gid):
                return make_cable_cell(gid, self.params)

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

            def get_probes(self, id):
                # Probe just the membrane voltage at a location on the soma.
                return [arbor.cable_probe_membrane_voltage('(location 0 0)')]
