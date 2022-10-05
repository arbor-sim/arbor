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
        arb::domain_decomposition decomp = arb::partition_load_balance(recipe, context);

        // Instantiate the simulation.
        arb::simulation sim(recipe, decomp, context);

All the simulation's constructor arguments are optional, except the recipe, and assume
default values if not specified. In order to simplify construction of a simulation, the helper class
:cpp:class:`arb::simulation_builder` can be used to better control constrution arguments:

.. container:: example-code

    .. code-block:: cpp

        arb::simulation sim =                // implicit conversion to simulation
            arb::simulation::create(recipe)  // the recipe is always required
                .add_context(context)        // optionally add a context
                .add_decompostion(decomp)    // optionally add a decompostion
                .add_seed(42);               // optionally add a seed value

Class documentation
-------------------

.. cpp:namespace:: arb

.. cpp:class:: simulation

    The executable form of a model. A simulation is constructed
    from a recipe, and then used to update and monitor model state.

    Simulations take the following inputs:

        * The **constructor** takes:

            *   an :cpp:class:`arb::recipe` that describes the model;
            *   an :cpp:class:`arb::domain_decomposition` that describes how the
                cells in the model are assigned to hardware resources;
            *   an :cpp:class:`arb::context` which is used to execute the simulation.
            *   a :cpp:class:`uint64_t` in order to seed the pseudo pandom number generator (optional)
        * **Experimental inputs** that can change between model runs, such
          as external spike trains.

    Simulations provide an interface for executing and interacting with the model:

        * **Advance model state** from one time to another and reset model
          state to its original state before simulation was started.
        * **I/O** interface for sampling simulation state during execution
          (e.g. voltage and current) and spike output.

    **Types:**

    .. cpp:type:: spike_export_function = std::function<void(const std::vector<spike>&)>

        User-supplied callback function used as a sink for spikes generated
        during a simulation. See :cpp:func:`set_local_spike_callback` and
        :cpp:func:`set_global_spike_callback`.

    **Constructor:**

    .. cpp:function:: simulation(const recipe& rec, const domain_decomposition& decomp, const context& ctx, std::uint64_t seed)

    **Static member functions:**

    .. cpp:function:: simulation_builder create(const recipe& rec)

        Returns a builder object to which the constructor arguments can be passed selectively (see
        also example above).

    **Experimental inputs:**

    .. cpp:function:: void inject_events(const pse_vector& events)

        Add events directly to targets.
        Must be called before calling :cpp:func:`run`, and must contain events that
        are to be delivered at or after the current simulation time.

    **Updating Model State:**

    .. cpp:function:: void reset()

        Reset the state of the simulation to its initial state.

    .. cpp:function:: time_type run(time_type tfinal, time_type dt)

        Run the simulation from current simulation time to :cpp:any:`tfinal`,
        with maximum time step size :cpp:any:`dt`.

    .. cpp:function:: void set_binning_policy(binning_kind policy, time_type bin_interval)

        Set event binning policy on all our groups.

    **I/O:**

    .. cpp:function:: sampler_association_handle add_sampler(\
                        cell_member_predicate probeset_ids,\
                        schedule sched,\
                        sampler_function f,\
                        sampling_policy policy = sampling_policy::lax)

        Note: sampler functions may be invoked from a different thread than that
        which called :cpp:func:`run`.

        (see the :ref:`sampling_api` documentation.)

    .. cpp:function:: void remove_sampler(sampler_association_handle)

        Remove a sampler.
        (see the :ref:`sampling_api` documentation.)

    .. cpp:function:: void remove_all_samplers()

        Remove all samplers from probes.
        (see the :ref:`sampling_api` documentation.)

    .. cpp:function:: std::size_t num_spikes() const

        The total number of spikes generated since either construction or
        the last call to :cpp:func:`reset`.

    .. cpp:function:: void set_global_spike_callback(spike_export_function export_callback)

        Register a callback that will periodically be passed a vector with all of
        the spikes generated over all domains (the global spike vector) since
        the last call.
        Will be called on the MPI rank/domain with id 0.

    .. cpp:function:: void set_local_spike_callback(spike_export_function export_callback)

        Register a callback that will periodically be passed a vector with all of
        the spikes generated on the local domain (the local spike vector) since
        the last call.
        Will be called on each MPI rank/domain with a copy of the local spikes.
