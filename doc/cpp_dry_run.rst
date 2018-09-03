.. _cppdistcontext:

Dry-run Mode
===================

Dry-run mode is used to simulate the performance of running an MPI distributed simulation
without having access to an HPC cluster or even MPI support. It is verifiable against an MPI
run with the same parameters. In dry-run mode, we describe the model on a single domain and
translate it to however many domains we want to simulate. This allows us to know the exact
behavior of the entire system by only running the simulation on a single node.
To support dry-run mode we use the following classes:

.. cpp:namespace:: arb

.. cpp:class:: dry_run_context

    Implements the :cpp:class:`arb::distributed_context` interface for
    fake-distributed computation.

    **Constructor:**

    .. cpp:function:: dry_run_context_impl(unsigned num_ranks, unsigned num_cells_per_tile)

        Sets up the dry run context with the information needed to fake communication between
        domains. :cpp:var:`num_ranks` is the total number of domains we are simulating; and
        :cpp:var:`num_cells_per_tile` is the number of cells assigned to each domain.

    **Interface:**

    .. cpp:function:: int id() const

        Always 0. We are only performing the simulation on one of the domains which will be root.

    .. cpp:function:: int size() const

        Equal to :cpp:var:`num_ranks`.

    .. cpp:function:: std::string name() const

        Returns ``"dry_run"``.

    .. cpp:function:: std::vector<std::string> gather(std::string value, int root) const

        Duplicates the vector of strings from domain :cpp:var:`root`, :cpp:var:`num-ranks` times.
        Returns the concatenated vector.

    .. cpp:function:: gathered_vector<arb::spike>  gather_spikes(const std::vector<arb::spike>& local_spikes) const

        The vector of local_spikes represents the spikes obtained from running a simulation of
        :cpp:var:`num_cells_per_tile` on the local domain. The function is used to gather the
        spikes obtained from all simulated domains. This is done by modifying the gids of each
        spike in local_spikes to refer to the corresponding gids on each simulated domain. The
        modified_local_spikes from each domain are concatenated with the original local_spikes
        and returned.

    .. cpp:function:: distributed_context_handle make_dry_run_context(unsigned num_ranks, unsigned num_cells_per_tile)

        Convenience function that returns a handle to a dry_run context.

.. cpp:class:: tile: public recipe

    While this class inherits from recipe, it breaks one important implicit rule of :cpp:class:`arb::recipe`:
    it allows connection from gids greater than the total number of cells in a recipe.
    `tile` describes the model on a single domain, which is to be duplicated over :cpp:var:`num_ranks` in
    dry-run mode. It contains information about :cpp:var:`num_ranks` which is provided by the following function:

    .. cpp:function:: :virtual cell_size_type num_tiles() const

    All overloaded functions in `tile` should describe a recipe on a singe domain much like we would describe a
    regular :cpp:class:`arb::recipe`. The exception is the following 2 functions:

    .. cpp:function:: virtual std::vector<cell_connection> connections_on(cell_gid_type i) const

    We allow connections from gids outside our domain. This is in order to create a realistic network.

    .. cpp:function:: virtual std::vector<event_generator> event_generators(cell_gid_type i) const

    We describe events on all gids from all domains.


.. cpp:class:: symmetric_recipe: public recipe

    A symmetric recipe simulates having a model containing cpp:function:`arb::tile::num_tiles`
    instances of :cpp:class:`arb::tile`.

    .. cpp:type:: std::unique_ptr<tile> tiled_recipe_

    `symmetric_recipe` owns a unique pointer to a :cpp:class:`arb::tile`, and uses the `tiled_recipe_`
    to query information about the tiles on the simulated domains.

    Most functions in `symmetric_recipe` only need to call the underlying functions of `tiled_recipe_` for the
    corresponding gid in the simulated domain. This is done with a simple modulo operation. For example:
    `get_cell_description(i)` calls `tiled_recipe_.get_cell_description(i % tiled_recipe_->num_cells())`.

    The exception is again the following 2 functions:

    .. cpp:function:: virtual std::vector<cell_connection> connections_on(cell_gid_type i) const

    Calls `tiled_recipe_.connections_on(i % tiled_recipe_->num_cells())`. But the connections have to
    be translated to refer to the correct gids corresponding to the correct domain.

    .. cpp:function:: virtual std::vector<event_generator> event_generators(cell_gid_type i) const

    Calls `tiled_recipe_.event_generators(i)`. We call on the correct domain gid immediately.





