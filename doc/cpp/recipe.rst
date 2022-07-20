.. _cpprecipe:

Recipes
===============

The :cpp:class:`arb::recipe` class documentation is below.

.. _cpp_recipe_best_practice:

C++ best practices
------------------

Here we collect rules of thumb to keep in mind when making recipes in C++.

.. topic:: Stay thread safe

    The load balancing and model construction are multithreaded, that is
    multiple threads query the recipe simultaneously.
    Hence calls to a recipe member should not have side effects, and should use
    lazy evaluation when possible (see `Be lazy <_recipe_lazy>`_).


Recipe
------

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
        Each connection ``con`` should have a valid synapse label ``con.dest`` on the post-synaptic target `gid`,
        and a valid source label ``con.source.label`` on the pre-synaptic source ``con.source.gid``.
        See :cpp:type:`cell_connection`.

        By default returns an empty list.

    .. cpp:function:: virtual std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const

        Returns a list of all the gap junctions connected to `gid`.
        Each gap junction ``gj`` should have a valid gap junction site label ``gj.local`` on ``gid``,
        and a valid gap junction site label ``gj.peer.label`` on ``gj.peer.gid``.
        See :cpp:type:`gap_junction_connection`.

        By default returns an empty list.

    .. cpp:function:: virtual std::vector<event_generator> event_generators(cell_gid_type gid) const

        Returns a list of all the event generators that are attached to `gid`.

        By default returns an empty list.

    .. cpp:function:: virtual std::vector<probe_info> get_probes(cell_gid_type gid) const

        Intended for use by cell group implementations to set up sampling data
        structures ahead of time and for putting in place any structures or
        information in the concrete cell implementations to allow monitoring.

        Returns a vector containing (in order) all the probes on a given cell `gid`.

        By default returns an empty vector.

    .. cpp:function:: virtual std::any get_global_properties(cell_kind) const

        Global property type will be specific to given cell kind.

        By default returns an empty container.

Cells
--------

See :ref:`cppcell`.

Synapses
--------

See :ref:`cppinterconnectivity`.

Probes
------

.. cpp:type:: probe_tag = int

    Extra contextual information associated with a probe.

.. cpp:class:: probe_info

    Probes are specified in the recipe objects that are used to initialize a
    model; the specification of the item or value that is subjected to a
    probe will be specific to a particular cell type.

    .. cpp:member:: probe_tag tag

           Opaque key, returned in sample record.

    .. cpp:member:: util::any address

           Cell-type specific location info, specific to cell kind of ``id.gid``.

Event generator and schedules
-----------------------------


Example
-------
