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

By default, spikes are exchanged between all ranks with a global all-to-all after every
epoch, where an epoch is half the minimum delay of the whole network. If connections
between distant ranks have longer delays than those between neighbouring ranks, MPI
neighbourhood collectives can make this cheaper. Each rank names the ranks it receives
spikes from, grouped into levels. Arbor then splits the senders of each rank into

* the rank itself, which is handled without communication,
* one level per neighbourhood, exchanged with a sparse neighbourhood all-gather,
* all remaining (far) ranks, exchanged with a global all-to-all.

Spikes are buffered per level, and each level is only exchanged as often as the minimum
delay of the connections from it allows. All buffers are flushed at the end of each
:cpp:func:`simulation::run`.

.. container:: example-code

    .. code-block:: cpp

        // Receive from the neighbouring ranks on a ring of n ranks, then from the next but one.
        auto ring = [&](int offset) { return (rank + offset + n) % n; };
        arb::neighborhood_spike_exchange exchange;
        exchange.neighborhoods = {{ring(-1), ring(1)}, {ring(-2), ring(2)}};

        arb::simulation sim = arb::simulation::create(recipe)
            .set_context(context)
            .set_spike_exchange(exchange);

A rank listed in more than one level belongs to the first, and the calling rank itself is
ignored, so nested levels are fine. All ranks must pass the same number of levels. With
no levels at all, only the far ranks remain, which still saves collectives if delays
between ranks exceed the minimum delay.

Within a neighbourhood, spikes are exchanged with ``MPI_Neighbor_allgatherv``, which sends
every spike with targets in the neighbourhood to all of its ranks. The example in
``example/neighborhood`` compares this with the default exchange.

In a dry run, pass the levels of rank 0. Since every rank holds a copy of rank 0's tile
shifted by its rank, the levels of rank r are taken to be those of rank 0 shifted by r.

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
            *   a :cpp:class:`arb::neighborhood_spike_exchange` to exchange spikes with MPI
                neighbourhood collectives (optional, see above)
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
