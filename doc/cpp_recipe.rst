``arb::recipe``
===============

**intro**

Models are described using recipes.

Recipe -> model build -> model execution

Recipe provides interface that the model building phase queries to find out information on cells, such as:
* The number of cells in the model
* Type of each cell
* A description of each cell
* network connections on each cell

Recipe:

* cell-centric view: a recipe can be queried for model properties for each cell
* recipe can be evaluated lazily
* parallel partition of recipe parsing and model generation

Recipes are highly portable, because they only describe the model, with no
information about how the model might be represented or simulated.

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


