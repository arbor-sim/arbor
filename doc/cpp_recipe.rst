``arb::recipe``
===============

A recipe is a description of a model, describing the cells and network, without any information about
how the model is to be executed.

An Arbor recipe provides interface that the model building phase queries to
find out information on cells, such as:

  * the number of cells in the model;
  * type of a cell;
  * a description of a cell;
  * incoming network connections on a cell.

.. cpp:namespace:: arb

.. cpp:class:: recipe

    A recipe is a description of a model, describing the cells and network, without any
    information about how the model is to be represented or executed.

    All recipes derive from :cpp:type:`arb::recipe`, and abstract base class in ``src/recipe.hpp``.

    .. Warning::
        All member functions must be **thread safe**, because the recipe is used by
        multi-threaded code.

    **Required Member Functions**

    The following member functions are required:

    .. cpp:function:: virtual cell_size_type num_cells() const

        The number of cells in the model.

    .. cpp:function:: virtual cell_kind get_cell_kind(cell_gid_type gid) const

        The kind of the cell with :cpp:var:`gid`. :cpp:type:`arb::cell_kind` is an enum
        with possible values:

    .. cpp:function:: virtual util::unique_any get_cell_description(cell_gid_type gid) const

        A description of the cell with :cpp:var:`gid`.
        The type used to *describe* a cell depends on the kind of the cell.
        The interface for querying the kind and description of a cell are
        seperate to optimize model building: cell descriptions can be expensive
        to generate, e.g. a description of the morphology and ion channels plus
        synapses of a multi-compartment cell.

    **Optional Member Functions**

    .. cpp:function:: virtual foo() const

        description of foo

    .. cpp:function:: virtual foo() const

        description of foo

.. code-block:: C++

    // required
    virtual cell_size_type num_cells() const = 0;
    virtual util::unique_any get_cell_description(cell_gid_type gid) const = 0;
    virtual cell_kind get_cell_kind(cell_gid_type) const = 0;

    // optional
    virtual cell_size_type num_sources(cell_gid_type) const { return 0; }
    virtual cell_size_type num_targets(cell_gid_type) const { return 0; }
    virtual cell_size_type num_probes(cell_gid_type)  const { return 0; }
    virtual std::vector<event_generator> event_generators(cell_gid_type) const { return {}; }
    virtual std::vector<cell_connection> connections_on(cell_gid_type) const { return {}; }
    virtual probe_info get_probe(cell_member_type) const { throw std::logic_error("no probes"); }
    // Global property type will be specific to given cell kind.
    virtual util::any get_global_properties(cell_kind) const { return util::any{}; };


Recipe Best Practices
---------------------

Recipe:

* cell-centric view: a recipe can be queried for model properties for each cell
* recipe can be evaluated lazily
* parallel partition of recipe parsing and model generation

Recipes are highly portable, because they only describe the model, with no
information about how the model might be represented or simulated.

