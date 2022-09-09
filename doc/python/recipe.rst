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
    This means that model properties, such as connections, are queried using the global identifier ``gid`` of a cell.
    In the description below, the term ``gid`` is used as shorthand for the cell with global identifier.

    **Required Constructor**

    The constructor must be implemented and call the base class constructor, as at the moment there is no way
    to instruct Python to do that automatically.
    
    .. note:: 
        Arbor's Python binding is that: a thin wrappper around the Arbor library which is written in C++.
        Calling the base class constructor ensures correct initialization of memory in the underlying C++ class.
    
    A minimal constructor therefore looks like this:

    .. code-block:: python

        def __init__(self):
            arbor.recipe.__init__(self)

    **Required Member Functions**

    The following member functions (besides a constructor) must be implemented by every recipe:

    .. function:: num_cells()

        The number of cells in the model.

    .. function:: cell_kind(gid)

        The cell kind of the cell with global identifier ``gid`` (return type: :class:`arbor.cell_kind`).

    .. function:: cell_description(gid)

        A high level description of the cell with global identifier ``gid``,
        for example the morphology, synapses and ion channels required to build a multi-compartment neuron.
        The type used to describe a cell depends on the kind of the cell.
        The interface for querying the kind and description of a cell are separate
        to allow the cell type to be provided without building a full cell description,
        which can be very expensive.

    **Optional Member Functions**

    .. function:: connections_on(gid)

        Returns a list of all the **incoming** connections to ``gid``.
        Each connection should have a valid synapse label ``connection.dest`` on the post-synaptic target ``gid``,
        and a valid source label ``connection.source.label`` on the pre-synaptic source ``connection.source.gid``.
        See :class:`connection`.

        By default returns an empty list.

    .. function:: gap_junctions_on(gid)

        Returns a list of all the gap junctions connected to ``gid``.
        Each gap junction ``gj`` should have a valid gap junction site label ``gj.local`` on ``gid``,
        and a valid gap junction site label ``gj.peer.label`` on ``gj.peer.gid``.
        See :class:`gap_junction_connection`.

        By default returns an empty list.

    .. function:: event_generators(gid)

        A list of all the :class:`event_generator` s that are attached to ``gid``.

        By default returns an empty list.

    .. function:: probes(gid)

        Returns a list specifying the probesets describing probes on the cell ``gid``.
        Each element in the list is an opaque object of type :class:`probe` produced by
        cell kind-specific probeset functions. Each probeset in the list
        has a corresponding probeset id of type :class:`cell_member`: an id ``(gid, i)``
        refers to the probes described by the ith entry in the list returned by ``get_probes(gid)``.

        By default returns an empty list.

    .. function:: global_properties(kind)

        The global properties of a model.

        This method needs to be implemented for :class:`arbor.cell_kind.cable`, where the
        properties include ion concentrations and reversal potentials; initial membrane voltage;
        temperature; axial resistivity; membrane capacitance; cv_policy; and a pointer
        to the mechanism catalogue. Also see :ref:`mechanisms_builtins`.

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

        Construct an event generator for a :attr:`target` synapse with :attr:`weight` of the events to
        deliver based on a schedule (i.e., :class:`arbor.regular_schedule`, :class:`arbor.explicit_schedule`,
        :class:`arbor.poisson_schedule`).

    .. attribute:: target

        The target synapse of type :class:`arbor.cell_local_label`.

    .. attribute:: weight

        The weight delivered to the target synapse. It is up to the target mechanism to interpret this quantity.
        For Arbor-supplied point processes, such as the ``expsyn`` synapse, a weight of ``1`` corresponds to an
        increase in conductivity in the target mechanism of ``1`` μS (micro-Siemens).

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
        with an expected frequency :attr:`freq` = 10 kHz and :attr:`seed` = 0.

    .. attribute:: tstart

        The delivery time of the first event in the sequence [ms].

    .. attribute:: freq

        The expected frequency [kHz].

    .. attribute:: seed

        The seed for the random number generator.

    .. function:: events(t0, t1)

        Returns a view of monotonically increasing time values in the half-open interval [t0, t1).

    .. attribute:: tstop

        No events delivered after this time [ms].

An example of an event generator reads as follows:

.. container:: example-code

    .. code-block:: python

        import arbor

        # define a Poisson schedule with start time 1 ms, expected frequency of 5 Hz,
        # and the target cell's gid as seed
        def event_generators(gid):
            target = arbor.cell_local_label("syn", arbor.selection_policy.round_robin) # label of the synapse on target cell gid
            seed   = gid
            tstart = 1
            freq   = 0.005
            sched  = arbor.poisson_schedule(tstart, freq, seed)

            # construct an event generator with this schedule on target cell and weight 0.1
            w = 0.1
            return [arbor.event_generator(target, w, sched)]

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
                # Cell should have a synapse labeled "syn"
                # and a detector labeled "detector"
                return make_cable_cell(gid, self.params)

            # The kind method returns the type of cell with gid.
            # Note: this must agree with the type returned by cell_description.
            def cell_kind(self, gid):
                return arbor.cell_kind.cable

            # Make a ring network
            def connections_on(self, gid):
                src = (gid-1)%self.ncells
                w = 0.01
                d = 10
                return [arbor.connection((src,"detector"), "syn", w, d)]

            # Attach a generator to the first cell in the ring.
            def event_generators(self, gid):
                if gid==0:
                    sched = arbor.explicit_schedule([1])
                    return [arbor.event_generator("syn", 0.1, sched)]
                return []

            def get_probes(self, id):
                # Probe just the membrane voltage at a location on the soma.
                return [arbor.cable_probe_membrane_voltage('(location 0 0)')]
