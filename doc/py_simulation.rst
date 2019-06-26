.. _pysimulation:

Simulations
===========

From recipe to simulation
-------------------------

To build a simulation the following concepts are needed:

    * an :class:`arbor.recipe` that describes the cells and connections in the model;
    * an :class:`arbor.context` used to execute the simulation.

The workflow to build a simulation is to first generate an
:class:`arbor.domain_decomposition` based on the :class:`arbor.recipe` and :class:`arbor.context` describing the distribution of the model
over the local and distributed hardware resources (see :ref:`pydomdec`). Then, the simulation is build using this :class:`arbor.domain_decomposition`.

.. container:: example-code

    .. code-block:: python

        import arbor

        # Get a communication context (with 4 threads, no GPU)
        context = arbor.context(threads=4, gpu_id=None)

        # Initialise a recipe of user defined type my_recipe with 100 cells.
        n_cells = 100
        recipe = my_recipe(n_cells)

        # Get a description of the partition of the model over the cores.
        decomp = arbor.partition_load_balance(recipe, context)

        # Instatitate the simulation.
        sim = arbor.simulation(recipe, decomp, context)

        # Run the simulation for 2000 ms with time stepping of 0.025 ms
        tSim = 2000
        dt = 0.025
        sim.run(tSim, dt)

.. currentmodule:: arbor

.. class:: simulation

    The executable form of a model.
    A simulation is constructed from a recipe, and then used to update and monitor the model state.

    Simulations take the following inputs:

    The **constructor** takes

        * an :class:`arbor.recipe` that describes the model;
        * an :class:`arbor.domain_decomposition` that describes how the cells in the model are assigned to hardware resources;
        * an :class:`arbor.context` which is used to execute the simulation.

    Simulations provide an interface for executing and interacting with the model:

        * **Advance the model state** from one time to another and reset the model state to its original state before simulation was started.
        * Sample the simulation state during the execution (e.g. compartment voltage and current) and generate spike output by using an **I/O interface**.

    **Constructor:**

    .. function:: simulation(recipe, domain_decomposition, context)

        Initialize the model described by an :class:`arbor.recipe`, with cells and network distributed according to :class:`arbor.domain_decomposition`, and computational resources described by :class:`arbor.context`.

    **Updating Model State:**

    .. function:: reset()

        Reset the state of the simulation to its initial state.

    .. function:: run(tfinal, dt)

        Run the simulation from current simulation time to ``tfinal``,
        with maximum time step size ``dt``.

        :param tfinal: The final simulation time [ms].

        :param dt: The time step size [ms].

    .. function:: set_binning_policy(policy, bin_interval)

        Set the binning ``policy`` for event delivery, and the binning time interval ``bin_interval`` if applicable [ms].

        :param policy: The binning policy of type :class:`binning`.

        :param bin_interval: The binning time interval [ms].

    **Types:**

    .. class:: binning

        Enumeration for event time binning policy.

        .. attribute:: none

            No binning policy.

        .. attribute:: regular

            Round time down to multiple of binning interval.

        .. attribute:: following

            Round times down to previous event if within binning interval.

Recording spikes
----------------
In order to analyze the simulation output spikes can be recorded.

**Types**:

.. class:: spike

    .. function:: spike()

        Construct a spike.

    .. attribute:: source

        The spike source (type: :class:`arbor.cell_member`).

    .. attribute:: time

        The spike time [ms].

.. class:: spike_recorder

    .. function:: spike_recorder()

        Initialize the spike recorder.

    .. attribute:: spikes

        The recorded spikes (type: :class:`spike`).

**I/O interface**:

.. function:: attach_spike_recorder(sim)

       Attach a spike recorder to an arbor :class:`simulation` ``sim``.
       The recorder that is returned will record all spikes generated after it has been
       attached (spikes generated before attaching are not recorded).

.. container:: example-code

    .. code-block:: python

        import arbor

        # Instatitate the simulation.
        sim = arbor.simulation(recipe, decomp, context)

        # Build the spike recorder
        recorder = arbor.attach_spike_recorder(sim)

        # Run the simulation for 2000 ms with time stepping of 0.025 ms
        tSim = 2000
        dt = 0.025
        sim.run(tSim, dt)

        # Print the spikes and according spike time
        for s in recorder.spikes:
            print(s)

>>> <arbor.spike: source (0,0), time 2.15168 ms>
>>> <arbor.spike: source (1,0), time 14.5235 ms>
>>> <arbor.spike: source (2,0), time 26.9051 ms>
>>> <arbor.spike: source (3,0), time 39.4083 ms>
>>> <arbor.spike: source (4,0), time 51.9081 ms>
>>> <arbor.spike: source (5,0), time 64.2902 ms>
>>> <arbor.spike: source (6,0), time 76.7706 ms>
>>> <arbor.spike: source (7,0), time 89.1529 ms>
>>> <arbor.spike: source (8,0), time 101.641 ms>
>>> <arbor.spike: source (9,0), time 114.125 ms>
