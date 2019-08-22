.. _cpprecipe:

Recipes
===============

The :cpp:class:`arb::recipe` class documentation is below.

.. _cpp_recipe_best_practice:

C++ Best Practices
------------------

Here we collect rules of thumb to keep in mind when making recipes in C++.

.. topic:: Stay thread safe

    The load balancing and model construction are multithreaded, that is
    multiple threads query the recipe simultaneously.
    Hence calls to a recipe member should not have side effects, and should use
    lazy evaluation when possible (see `Be lazy <_recipe_lazy>`_).


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
        by the multithreaded model building stage. In practice, this means that
        multiple threads should be able to call member functions of a recipe
        simultaneously. Model building is multithreaded to reduce model building times,
        so recipe implementations should avoid using locks and mutexes to introduce
        thread safety. See `recipe best practices <cpp_recipe_best_practice_>`_ for more
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
        separate to allow the cell type to be provided without building
        a full cell description, which can be very expensive.

    **Optional Member Functions**

    .. cpp:function:: virtual std::vector<cell_connection> connections_on(cell_gid_type gid) const

        Returns a list of all the **incoming** connections for `gid` .
        Each connection ``con`` should have post-synaptic target ``con.dest.gid`` that matches
        the argument :cpp:any:`gid`, and a valid synapse id ``con.dest.index`` on `gid`.
        See :cpp:type:`cell_connection`.

        By default returns an empty list.

    .. cpp:function:: virtual std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const

        Returns a list of all the gap junctions connected to `gid`.
        Each gap junction ``gj`` should have one of the two gap junction sites ``gj.local.gid`` or
        ``gj.peer.gid`` matching the argument :cpp:any:`gid`, and the corresponding synapse id
        ``gj.local.index`` or ``gj.peer.index`` should be valid on `gid`.
        See :cpp:type:`gap_junction_connection`.

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

    .. cpp:function:: virtual cell_size_type num_gap_junction_sites(cell_gid_type gid) const

        Returns the number of gap junction sites on `gid`.

        By default returns 0.

    .. cpp:function:: virtual probe_info get_probe(cell_member_type) const

        Intended for use by cell group implementations to set up sampling data
        structures ahead of time and for putting in place any structures or
        information in the concrete cell implementations to allow monitoring.

        By default throws :cpp:type:`std::logic_error`. If :cpp:func:`num_probes`
        returns a non-zero value, this must also be overridden.

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

.. cpp:class:: gap_junction_connection

    Describes a gap junction between two gap junction sites.
    Gap junction sites are represented by :cpp:type:cell_member_type.

    .. cpp:member:: cell_member_type local

        gap junction site: one half of the gap junction connection.

    .. cpp:member:: cell_member_type peer

        gap junction site: other half of the gap junction connection.

    .. cpp:member:: float ggap

        gap junction conductance in μS.
