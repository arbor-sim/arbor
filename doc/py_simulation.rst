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

Recording samples
-----------------

Definitions
***********

probe
    A location or component of a cell that is available for monitoring (see :attr:`arbor.recipe.num_probes`, :attr:`arbor.recipe.get_probe` and :attr:`arbor.cable_probe` as references).

sample/record
    A record of data corresponding to the value at a specific *probe* at a specific time.

sampler/sample recorder
    A function that receives a sequence of *sample* records.

Samples and sample recorders
****************************

In order to analyze the data collected from an :class:`arbor.probe` the samples can be recorded.

**Types**:

.. class:: sample

    .. attribute:: time

        The sample time [ms] at a specific probe.

    .. attribute:: value

        The sample record at a specific probe.

.. class:: sampler

    .. function:: sampler()

        Initialize the sample recorder.

    .. function:: samples(probe_id)

        A list of the recorded samples of a probe with probe id.

**Sampling interface**:

.. function:: attach_sampler(sim, dt)

    Attach a sample recorder to an arbor simulation.
    The recorder will record all samples from a regular sampling interval [ms] (see :class:`arbor.regular_schedule`) matching all probe ids.

.. function:: attach_sampler(sim, dt, probe_id)

    Attach a sample recorder to an arbor simulation.
    The recorder will record all samples from a regular sampling interval [ms] (see :class:`arbor.regular_schedule`) matching one probe id.

.. container:: example-code

    .. code-block:: python

        import arbor

        # Instatitate the simulation.
        sim = arbor.simulation(recipe, decomp, context)

        # Build the sample recorder on cell 0 and probe 0 with regular sampling interval of 0.1 ms
        pid = arbor.cell_member(0,0) # cell 0, probe 0
        sampler = arbor.attach_sampler(sim, 0.1, pid)

        # Run the simulation for 100 ms
        sim.run(100)

        # Print the sample times and values
        for sa in sampler.samples(pid):
            print(sa)

>>> <arbor.sample: time 0 ms,       value -65>
>>> <arbor.sample: time 0.1 ms,     value -64.9981>
>>> <arbor.sample: time 0.2 ms,     value -64.9967>
>>> <arbor.sample: time 0.3 ms,     value -64.9956>
>>> <arbor.sample: time 0.4 ms,     value -64.9947>
>>> <arbor.sample: time 0.475 ms,   value -64.9941>
>>> <arbor.sample: time 0.6 ms,     value -64.9932>
>>> <arbor.sample: time 0.675 ms,   value -64.9927>
>>> <arbor.sample: time 0.8 ms,     value -64.992>
>>> <arbor.sample: time 0.9 ms,     value -64.9916>
>>> <arbor.sample: time 1 ms,       value -64.9912>
>>> <arbor.sample: time 1.1 ms,     value -62.936>
>>> <arbor.sample: time 1.2 ms,     value -59.2284>
>>> <arbor.sample: time 1.3 ms,     value -55.8485>
>>> <arbor.sample: time 1.375 ms,   value -53.663>
>>> <arbor.sample: time 1.475 ms,   value -51.0649>
>>> <arbor.sample: time 1.6 ms,     value -47.9543>
>>> <arbor.sample: time 1.7 ms,     value -45.1928>
>>> <arbor.sample: time 1.8 ms,     value -41.7243>
>>> <arbor.sample: time 1.875 ms,   value -38.2573>
>>> <arbor.sample: time 1.975 ms,   value -31.576>
>>> <arbor.sample: time 2.1 ms,     value -17.2756>
>>> <arbor.sample: time 2.2 ms,     value 0.651031>
>>> <arbor.sample: time 2.275 ms,   value 15.0592>

