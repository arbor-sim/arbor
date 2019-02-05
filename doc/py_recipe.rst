.. _pyrecipe:

Recipes
=================

A **recipe** describes neuron models in a cell-oriented manner and supplies methods to provide cell information. Details on why Arbor uses recipes and general best practices can be found in :ref:`modelrecipe`.

.. module:: arbor

.. class:: recipe

    A description of a model, describing the cells and network, without any
    information about how the model is to be represented or executed.

    All recipes derive from this abstract base class.

    Recipes provide a cell-centric interface for describing a model. This means that
    model properties, such as connections, are queried using the global identifier
    (`gid`) of a cell. In the description below, the term `gid` is used as shorthand
    for the cell with global identifier `gid`.

    **Required Member Functions**

    The following member functions (besides a constructor) must be implemented by every recipe:

    .. function:: num_cells()

        The number of cells in the model.

    .. function:: cell_kind(gid)

        The cell kind of cell with global identifier gid (type: :class:`arbor.cell_kind`).

    .. function:: cell_description(gid)

        High level decription of the cell with global identifier gid,
        for example the morphology, synapses and ion channels required to build a multi-compartment neuron.
        The type used to describe a cell depends on the kind of the cell.
        The interface for querying the kind and description of a cell are seperate
        to allow the cell type to be provided without building a full cell description,
        which can be very expensive.

    **Optional Member Functions**

    .. function:: num_sources(gid)

        The number of spike sources on gid.

    .. function:: num_targets(gid)

        The number of event targets on gid (e.g. synapses).

    .. function:: connections_on(gid)

        Returns a list of all the incoming connections for `gid`.
        Each connection should have post-synaptic target :attr:`arbor.connection.destination` that matches the argument :attr:`arbor.cell_member.gid`, and a valid synapse id :attr:`arbor.cell_member.index` on `gid`.
        See :class:`arbor.connection`.

        By default returns an empty list.

    .. function:: event_generator(index, weight, schedule)

        Returns a list of all the event generators that are attached to the gid with cell-local index, weight and schedule :class:`regular_schedule`, :class:`explicit_schedule` or :class:`poisson_schedule`.

        By default returns an empty list.


.. class:: connection

    Describes a connection between two cells: a pre-synaptic source and a post-synaptic destination.
    The source is typically a threshold detector on a cell or a spike source.
    The destination is a synapse on the post-synaptic cell.

    .. function:: connection(source, destination, weight, delay)

        Constructor.

    .. attribute:: source

        The source of the connection (type: :class:`arbor.cell_member`).

    .. attribute:: destination

        The destination of the connection (type: :class:`arbor.cell_member`).

    .. attribute:: weight

        The weight of the connection (S⋅cm⁻²).

    .. attribute:: delay

        The delay time of the connection (ms).

.. class:: regular_schedule

    .. function:: regular_schedule()

        Default constructor with empty time range and zero time step size.

    .. function:: regular_schedule(tstart, tstop, dt)

        Constructor creates a list of times from :attr:`tstart` to :attr:`tstop` in :attr:`dt` time steps.

    .. attribute:: tstart

        Start time (in ms).

    .. attribute:: tstop

        End time (in ms).

    .. attribute:: dt

        Time step size (in ms).


.. class:: explicit_schedule

    .. function:: explicit_schedule()

        Constructor with empty list.

    .. attribute:: times

        A list of times in the schedule (in ms).

.. class:: poisson_schedule

    To be implemented.


Cells
------
A multicompartmental cell in Arbor's python front end can be created by making a soma and adding synapses at specific segment locations.

.. class:: make_soma_cell

    Make a single compartment cell with properties:

    - diameter 18.8 µm;
    - Hodgkin-Huxley (HH) mechanisms (with default parameters);
    - bulk resistivitiy 100 Ω·cm (default);
    - capacitance 0.01 F⋅m⁻² (default).

    The default parameters of HH mechanisms are:

    - Na-conductance 0.12 S⋅m⁻²,
    - K-conductance 0.036 S⋅m⁻²,
    - passive conductance 0.0003 S⋅m⁻² and
    - passive potential -54.3 mV

 .. class:: segment_location(segment, position)

    Sets the location to a cell-local segment and a position.

        .. attribute:: segment

            Segment as cell-local index.

        .. attribute:: position

            Position between 0 and 1.

.. class:: mccell

        .. function:: add_synapse(location)

            Add an exponential synapse at segment location.

        .. function:: add_stimulus(location, t0, duration, weight)

            Add a stimulus to the cell at a specific location, start time t0 (in ms), duration (in ms) with weight (in nA).

        .. function:: add_detector(location, threshold)

            Add a detector to the cell at a specific location and threshold (in mV).


An example of a recipe construction of a ring network of multicompartmental cells reads as follows:

.. container:: example-code

    .. code-block:: python

        import arbor

        # A recipe, that describes the cells and network of a model, can be defined
        # in python by implementing the arbor.recipe interface.

        class ring_recipe(arbor.recipe):

            def __init__(self, n=4):
                # The base C++ class constructor must be called first, to ensure that
                # all memory in the C++ class is initialized correctly.
                arbor.recipe.__init__(self)
                self.ncells = n

            # The num_cells method that returns the total number of cells in the model
            # must be implemented.
            def num_cells(self):
                return self.ncells

            # The cell_description method returns a cell
            def cell_description(self, gid):
                # Make a soma cell
                cell = arbor.make_soma_cell()

                # Add synapse at segment 0 at location 0.5
                loc = arbor.segment_location(0, 0.5)
                cell.add_synapse(loc)

                # Add stimulus to first cell with gid 0 at t0 = 0 ms for duration of 20 ms with weight 0.01 nA
                if gid==0:
                cell.add_stimulus(loc, 0, 20, 0.01)
                return cell

            def num_targets(self, gid):
                return 1

            def num_sources(self, gid):
                return 1

            # The kind method returns the type of cell with gid.
            # Note: this must agree with the type returned by cell_description.
            def kind(self, gid):
                return arbor.cell_kind.cable1d

            # Make a ring network
            def connections_on(self, gid):
                # Define the source of cell with gid as the previous cell with gid-1
                #    caution: close the ring at gid 0
                src = self.num_cells()-1 if gid==0 else gid-1
                return [arbor.connection(arbor.cell_member(src,0), arbor.cell_member(gid,0), 0.1, 10)]
