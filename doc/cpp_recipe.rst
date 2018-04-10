Recipes
===============

An Arbor recipe is a description of a model. The recipe is queried during the model
building phase to provide cell information, such as:

  * the number of cells in the model;
  * the type of a cell;
  * a description of a cell;
  * incoming network connections on a cell.

The :cpp:class:`arb::recipe` class documentation is below.

Why Recipes?
--------------

The interface and design of Arbor recipes was motivated by the following aims:

    * Building a simulation from a recipe description must be possible in a
      distributed system efficiently with minimal communication.
    * To minimise the amount of memory used in model building, to make it
      possible to build and run simulations in one run.

Recipe descriptions are cell-oriented, in order that the building phase can
be efficiently distributed and that the model can be built independently of any
runtime execution environment.

During model building, the recipe is queried first by a load balancer,
then later when building the low-level cell groups and communication network.
The cell-centered recipe interface, whereby cell and network properties are
specified "per-cell", facilitates this.

The steps of building a simulation from a recipe are:

.. topic:: 1. Load balancing

    First, the cells are partitioned over MPI ranks, and each rank parses
    the cells assigned to it to build a cost model.
    The ranks then coordinate to redistribute cells over MPI ranks so that
    each rank has a balanced workload. Finally, each rank groups its local
    cells into :cpp:type:`cell_group` s that balance the work over threads (and
    GPU accelerators if available).

.. topic:: 2. Model building

    The model building phase takes the cells assigned to the local rank, and builds the
    local cell groups and the part of the communication network by querying the recipe
    for more information about the cells assigned to it.

.. _recipe_best_practice:

Best Practices
--------------

Here is a set of rules of tumb to keep in mind when making recipes. The first is
mandatory, and following the others as closely as possible will lead to better
performance.

.. topic:: Stay thread safe

    The load balancing and model construction are multithreaded, that is
    multiple threads query the recipe simultaneously.
    Hence calls to a recipe member should not have side effects, and should use
    lazy evaluation when possible (see `Be lazy <recipe_lazy_>`_).

.. _recipe_lazy:

.. topic:: Be lazy

    A recipe does not have to contain a complete description of the model in
    memory; it should precompute as little as possible, and use
    `lazy evaluation <https://en.wikipedia.org/wiki/Lazy_evaluation>`_ to generate
    information only when requested.
    This has multiple benefits, including:

        * thread safety;
        * minimising memory footprint of recipe.

.. topic:: Think of the cells

    When formulating a model, think cell-first, and try to formulate the model and
    the associated workflow from a cell-centered perspective. If this isn't possible,
    please contact the developers, because we would like to develop tools that help
    make this simpler.

.. topic:: Be reproducible

    Arbor is designed to give reproduceable results when the same model is run on a
    different number of MPI ranks or threads, or on different hardware (e.g. GPUs).
    This only holds when a recipe provides a reproducible model description, which
    can be a challenge when a description uses random numbers, e.g. to pick incoming
    connections to a cell from a random subset of a cell population.
    To get a reproduceable model, use the cell `gid` (or a hash based on the `gid`)
    to seed random number generators, including those for :cpp:type:`event_generator` s.


Class Documentation
-------------------

.. cpp:namespace:: arb

.. cpp:class:: recipe

    A description of a model, describing the cells and network, without any
    information about how the model is to be represented or executed.

    All recipes derive from this abstract base class, defined in ``src/recipe.hpp``.

    Recipes provide a cell-centric interface for describing a model. This means that
    model properties, such as connections, are queried using the global identifier
    (`gid`) of a cell. In the description below, the term `gid` is used as shorthand
    for "the cell with global identifier `gid`".


    .. Warning::
        All member functions must be **thread safe**, because the recipe is used
        by the multithreaded model builing stage. In practice, this means that
        multiple threads should be able to call member functions of a recipe
        simultaneously. Model building is multithreaded to reduce model building times,
        so recipe implementations should avoid using locks and mutexes to introduce
        thread safety. See `recipe best practices <recipe_best_practice_>`_ for more
        information.

    **Required Member Functions**

    The following member functions must be implemented by every recipe:

    .. cpp:function:: virtual cell_size_type num_cells() const = 0

        The number of cells in the model.

    .. cpp:function:: virtual cell_kind get_cell_kind(cell_gid_type gid) const = 0

        The kind of `gid` (see :cpp:type:`arb::cell_kind`).

    .. cpp:function:: virtual util::unique_any get_cell_description(cell_gid_type gid) const = 0

        A description of the cell `gid`, for example the morphology, synapses
        and ion channels required to build a multi-compartment neuron.

        The type used to describe a cell depends on the kind of the cell.
        The interface for querying the kind and description of a cell are
        seperate to allow the the cell type to be provided without building
        a full cell description, which can be very expensive.

    **Optional Member Functions**

    .. cpp:function:: virtual std::vector<cell_connection> connections_on(cell_gid_type gid) const

        Returns a list of all the **incoming** connections for `gid` .
        Each connection ``con`` should have post-synaptic target ``con.dest.gid`` that matches
        the argument :cpp:var:`gid`, and a valid synapse id ``con.dest.index`` on `gid`.
        See :cpp:type:`cell_connection`.

        By default returns an empty list.

    .. cpp:function:: virtual std::vector<event_generator> event_generators(cell_gid_type gid) const

        Returns a list of all the event generators that are attached to `gid`.

        By default returns an empty list.

    .. cpp:function:: virtual cell_size_type num_sources(cell_gid_type gid) const

        Returns the number of spike sources on `gid`. This corresponds to the number
        of spike detectors on a multi-compartment cell. Typically there is one detector
        at the soma of the cell, however it is possible to attache multiple detectors
        at arbitrary locations.

        By default returns 0.

    .. cpp:function:: virtual cell_size_type num_targets(cell_gid_type gid) const

        The number of post-synaptic sites on `gid`, which corresponds to the number
        of synapses.

        By default returns 0.

    .. cpp:function:: virtual cell_size_type num_probes(cell_gid_type gid) const

        The number of probes attached to the cell.

        By default returns 0.

    .. cpp:function:: virtual probe_info get_probe(cell_member_type) const

        Intended for use by cell group implementations to set up sampling data
        structures ahead of time and for putting in place any structures or
        information in the concrete cell implementations to allow monitoring.

        By default throws :cpp:type:`std::logic_error`. If ``arb::recipe::num_probes``
        returns a non-zero value, this must also be overriden.

    .. cpp:function:: virtual util::any get_global_properties(cell_kind) const

        Global property type will be specific to given cell kind.

        By default returns an empty container.

.. cpp:class:: cell_connection

    Describes a connection between two cells: a pre-synaptic source and a
    post-synaptic destination. The source is typically a threshold detector on
    a cell or a spike source. The destination is a synapse on the post-synaptic cell.

    .. cpp:type:: cell_connection_endpoint = cell_member_type

        Connection end-points are represented by pairs
        (cell index, source/target index on cell).

    .. cpp:member:: cell_connection_endpoint source

        Source end point.

    .. cpp:member:: cell_connection_endpoint dest

        Destination end point.

    .. cpp:member:: float weight

        The weight delivered to the target synapse.
        The weight is dimensionless, and its interpretation is
        specific to the synapse type of the target. For example,
        the `expsyn` synapse interprets it as a conductance
        with units μS (micro-Siemens).

    .. cpp:member:: float delay

        Delay of the connection (milliseconds).

