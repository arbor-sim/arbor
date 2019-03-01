.. _cppdryrun:

.. Note::
    This is a developer feature for benchmarking, and is not useful for scientific use cases.

Dry-run Mode
===================

Dry-run mode is used to mimic the performance of running an MPI distributed simulation
without having access to an HPC cluster or even MPI support. It is verifiable against an MPI
run with the same parameters. In dry-run mode, we describe the model on a single domain and
translate it to however many domains we want to mimic. This allows us to know the exact
behavior of the entire system by only running the simulation on a single node.
To support dry-run mode we use the following classes:

.. cpp:namespace:: arb

.. cpp:class:: dry_run_context

    Implements the :cpp:class:`arb::distributed_context` interface for a fake distributed
    simulation.

    .. cpp:member:: unsigned num_ranks_

        Number of domains we are mimicking.

    .. cpp:member:: unsigned num_cells_per_tile_

        Number of cells assigned to each domain.


    **Constructor:**

    .. cpp:function:: dry_run_context_impl(unsigned num_ranks, unsigned num_cells_per_tile)

        Creates the dry run context and sets up the information needed to fake communication
        between domains.

    **Interface:**

    .. cpp:function:: int id() const

        Always 0. We are only performing the simulation on the local domain which will be root.

    .. cpp:function:: int size() const

        Equal to :cpp:member:`num_ranks_`.

    .. cpp:function:: std::string name() const

        Returns ``"dry_run"``.

    .. cpp:function:: std::vector<std::string> gather(std::string value, int root) const

        Duplicates the vector of strings from local domain, :cpp:member:`num_ranks_` times.
        Returns the concatenated vector.

    .. cpp:function:: gathered_vector<arb::spike>  gather_spikes(const std::vector<arb::spike>& local_spikes) const

        The vector of :cpp:any:`local_spikes` represents the spikes obtained from running a
        simulation of :cpp:member:`num_cells_per_tile_` on the local domain.
        The returned vector should contain the spikes obtained from all domains in the dry-run.
        The spikes from the non-simulated domains are obtained by copying :cpp:any:`local_spikes`
        and modifying the gids of each spike to refer to the corresponding gids on each domain.
        The obtained vectors of spikes from each domain are concatenated along with the original
        :cpp:any:`local_spikes` and returned.

    .. cpp:function:: distributed_context_handle make_dry_run_context(unsigned num_ranks, unsigned num_cells_per_tile)

        Convenience function that returns a handle to a :cpp:class:`dry_run_context`.

.. cpp:class:: tile: public recipe

    .. Note::
        While this class inherits from :cpp:class:`arb::recipe`, it breaks one of its implicit
        rules: it allows connection from gids greater than the total number of cells in a recipe,
        :cpp:any:`ncells`.

    :cpp:class:`arb::tile` describes the model on a single domain containing :cpp:expr:`num_cells =
    num_cells_per_tile` cells, which is to be duplicated over :cpp:any:`num_ranks`
    domains in dry-run mode. It contains information about :cpp:any:`num_ranks` which is provided
    by the following function:

    .. cpp:function:: cell_size_type num_tiles() const

    Most of the overloaded functions in :cpp:class:`arb::tile` describe a recipe on the local
    domain, as if it was the only domain in the simulation, except for the following two
    functions that accept :cpp:any:`gid` arguments in the half open interval
    ``[0, num_cells*num_tiles)``:

    .. cpp:function:: std::vector<cell_connection> connections_on(cell_gid_type gid) const

    .. cpp:function:: std::vector<event_generator> event_generators(cell_gid_type gid) const

.. cpp:class:: symmetric_recipe: public recipe

    A symmetric_recipe mimics having a model containing :cpp:var:`num_tiles()`
    instances of :cpp:class:`arb::tile` in a simulation of one tile per domain.

    .. cpp:member:: std::unique_ptr<tile> tiled_recipe_

        `symmetric_recipe` owns a unique pointer to a :cpp:class:`arb::tile`, and uses
        :cpp:member:`tiled_recipe_` to query information about the tiles on the local
        and mimicked domains.

        Most functions in `symmetric_recipe` only need to call the underlying functions
        of `tiled_recipe_` for the corresponding gid in the simulated domain. This is
        done with a simple modulo operation. For example:

        .. code-block:: cpp

            cell_kind get_cell_kind(cell_gid_type i) const override {
                return tiled_recipe_->get_cell_kind(i % tiled_recipe_->num_cells());
            }

    The exception is again the following 2 functions:

    .. cpp:function:: std::vector<cell_connection> connections_on(cell_gid_type i) const

        Calls

        .. code-block:: cpp

            tiled_recipe_.connections_on(i % tiled_recipe_->num_cells())

        But the obtained connections have to be translated to refer to the correct
        gids corresponding to the correct domain.

    .. cpp:function:: std::vector<event_generator> event_generators(cell_gid_type i) const

        Calls

        .. code-block:: cpp

            tiled_recipe_.event_generators(i)

        Calls on the domain gid without the modulo operation, because the function has a
        knowledge of the entire network.





