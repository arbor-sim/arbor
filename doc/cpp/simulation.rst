.. _cppsimulation:

Simulations
===========

From recipe to simulation
-------------------------

To build a simulation the following concepts are needed:

    * An :cpp:class:`arb::recipe` that describes the cells and connections
      in the model.
    * An :cpp:class:`arb::context` used to execute the simulation.

The workflow to build a simulation is to first generate a
:cpp:class:`arb::domain_decomposition` that describes the distribution of the model
over the local and distributed hardware resources (see :ref:`cppdomdec`),
then build the simulation.

.. container:: example-code

    .. code-block:: cpp

        #include <arbor/context.hpp>
        #include <arbor/domain_decomposition.hpp>
        #include <arbor/simulation.hpp>

        // Get a communication context
        arb::context context = make_context();

        // Make a recipe of user defined type my_recipe.
        my_recipe recipe;

        // Get a description of the partition the model over the cores
        // (and gpu if available) on node.
        auto decomp = arb::partition_load_balance(recipe, context);

        // Instantiate the simulation.
        arb::simulation sim(recipe, decomp, context);

All the simulation's constructor arguments are optional, except the recipe, and assume
default values if not specified. In order to simplify the construction of a simulation, the helper class
:cpp:class:`arb::simulation_builder` can be used to better control construction arguments:

.. container:: example-code

    .. code-block:: cpp

        arb::simulation sim =                // implicit conversion to simulation
            arb::simulation::create(recipe)  // the recipe is always required
                .add_context(context)        // optionally add a context
                .add_decompostion(decomp)    // optionally add a decompostion
                .add_seed(42);               // optionally add a seed value

Neighbourhood spike exchange
----------------------------

By default, spikes are exchanged between all ranks after every epoch. For spatially
structured networks, the :ref:`neighbourhood spike exchange <modelspikeexchange>` is often
much cheaper. It is a property of the :cpp:class:`context`, chosen when the context
is made with a :cpp:class:`neighborhood_spike_exchange`, and applies to all simulations on
that context. Each rank either lists the ranks it receives spikes from, grouped into
neighbourhoods:

.. container:: example-code

    .. code-block:: cpp

        // Receive from the neighbouring ranks on a ring of n ranks, then from the next but one.
        auto ring = [&](int offset) { return (rank + offset + n) % n; };
        arb::neighborhood_spike_exchange exchange{{{ring(-1), ring(1)}, {ring(-2), ring(2)}}};

        auto context = arb::make_context(resources, MPI_COMM_WORLD, exchange);
        arb::simulation sim = arb::simulation::create(recipe).set_context(context);

or which lets arbor choose the levels from the connection delays:

.. container:: example-code

    .. code-block:: cpp

        auto context = arb::make_context(resources, MPI_COMM_WORLD,
                                         arb::neighborhood_spike_exchange::automatic());

All ranks must choose the same kind of exchange; this is checked when a simulation is
built. Within a neighbourhood, spikes are
exchanged with ``MPI_Neighbor_allgatherv``, which sends every spike with targets in the
neighbourhood to all of its ranks; see the :ref:`developer documentation
<neighborhood_exchange>` for details.

:cpp:func:`simulation::spike_exchange_levels` returns the levels in use on the calling
rank, given or automatic, as a :cpp:class:`spike_exchange_level` each. The example in
``example/neighborhood`` prints them and compares the exchanges.

.. cpp:namespace:: arb

.. cpp:class:: neighborhood_spike_exchange

    How to exchange spikes between ranks; see :ref:`modelspikeexchange`.

    .. cpp:function:: explicit neighborhood_spike_exchange(std::vector<std::vector<int>> neighborhoods)

        The ranks this rank receives spikes from, grouped into neighbourhoods, innermost
        first. The neighbourhoods must be disjoint: no rank may be listed more than once. The
        calling rank is ignored. All ranks must pass the same number of neighbourhoods.

    .. cpp:function:: static neighborhood_spike_exchange off()

        The default exchange after every epoch.

    .. cpp:function:: static neighborhood_spike_exchange automatic()

        Choose the levels from the connection delays.

    .. cpp:function:: bool is_off() const

    .. cpp:function:: bool is_automatic() const

    .. cpp:function:: const std::vector<std::vector<int>>& neighborhoods() const

        The given neighbourhoods; empty if off or automatic.

.. cpp:class:: spike_exchange_level

    A level of the neighbourhood spike exchange, as seen by the calling rank.

    .. cpp:member:: std::vector<int> ranks

        The ranks this rank receives spikes from in this level, in ascending order.

    .. cpp:member:: std::size_t period

        The level is exchanged every ``period`` epochs of
        :cpp:func:`simulation::max_epoch_length`; 0 if no connections use it.

    .. cpp:member:: time_type min_delay

        The minimum delay [ms] of the connections in this level over all ranks, from which
        the period follows; infinity if there are none.

    .. cpp:member:: bool far

        Whether this is the far level, exchanged globally instead of within a neighbourhood.

Class documentation
-------------------

.. cpp:namespace:: arb

.. cpp:class:: simulation

    The executable form of a model. A simulation is constructed
    from a recipe, and then used to update and monitor the model state.

    Simulations take the following inputs:

        * The **constructor** takes:

            *   an :cpp:class:`arb::recipe` that describes the model;
            *   an :cpp:class:`arb::domain_decomposition` that describes how the
                cells in the model are assigned to hardware resources;
            *   an :cpp:class:`arb::context` which is used to execute the simulation.
            *   a :cpp:class:`std::uint64_t` in order to seed the pseudo random number generator (optional)
        * **Experimental inputs** that can change between model runs, such
          as external spike trains.

    Simulations provide an interface for executing and interacting with the model:

        * **Advance model state** from one time to another and reset model
          state to its original state before the simulation was started.
        * **I/O** interface for sampling simulation state during execution
          (e.g., voltage and current) and spike output.

    **Types:**

    .. cpp:type:: spike_export_function = std::function<void(const std::vector<spike>&)>

        User-supplied callback function used as a sink for spikes generated
        during a simulation. See :cpp:func:`set_local_spike_callback`.

    **Constructor:**

    .. cpp:function:: simulation(const recipe& rec, const domain_decomposition& decomp, const context& ctx, std::uint64_t seed)

    **Static member functions:**

    .. cpp:function:: simulation_builder create(const recipe& rec)

        Returns a builder object to which the constructor arguments can be passed selectively (see
        also example above).

    **Updating Model State:**

    .. cpp:function:: void reset()

        Reset the state of the simulation to its initial state.

    .. cpp:function:: time_type run(time_type tfinal, time_type dt)

        Run the simulation from the current simulation time to :cpp:any:`tfinal`,
        with maximum time step size :cpp:any:`dt`.

    **I/O:**

    .. cpp:function:: sampler_association_handle add_sampler(cell_member_predicate probeset_ids,\
                                                             schedule sched,\
                                                             sampler_function f)

        Note: sampler functions may be invoked from a different thread than that
        which is called :cpp:func:`run`.

        (see the :ref:`sampling_api` documentation.)

    .. cpp:function:: std::vector<probe_metadata> get_probe_metadata(const cell_address_type& probeset_id) const

       Return probe metadata, one entry per probe associated with the supplied probe
       id, or an empty vector if there is no local match for the probe id. See the
       :ref:`sampling_api` documentation.


    .. cpp:function:: void remove_sampler(sampler_association_handle)

        Remove a sampler.
        (see the :ref:`sampling_api` documentation.)

    .. cpp:function:: void remove_all_samplers()

        Remove all samplers from probes.
        (see the :ref:`sampling_api` documentation.)

    .. cpp:function:: std::size_t num_spikes() const

        The total number of spikes generated since either construction or
        the last call to :cpp:func:`reset`.

    .. cpp:function:: void set_local_spike_callback(spike_export_function export_callback)

        Register a callback that will periodically be passed a vector with all of
        the spikes generated on the local domain (the local spike vector) since
        the last call.
        Will be called on each MPI rank/domain with a copy of the local spikes.
