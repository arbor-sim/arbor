.. _pysimulation:

Simulations
===========

From recipe to simulation
-------------------------

To build a simulation the following are needed:

    * An :class:`arbor.recipe` that describes the cells and connections
      in the model.
    * An :class:`arbor.context` used to execute the simulation.

The workflow to build a simulation is to first generate an
:class:`arbor.domain_decomposition` that describes the distribution of the model
over the local and distributed hardware resources (see :ref:`pydomdec`),
then build the simulation.

.. container:: example-code

    .. code-block:: python

        import arbor

        # Get hardware resources, create a context
        resources = arbor.proc_allocation()
        context = arbor.context(resources)

        # Initialise a recipe of user defined type my_recipe with 100 cells.
        n_cells = 100
        recipe = my_recipe(n_cells)

        # Get a description of the partition the model over the cores
        # (and gpu if available) on node.
        decomp = arbor.partition_load_balance(recipe, context)

        # Instatitate the simulation.
        sim = arbor.simulation(recipe, decomp, context)

        # Run the simulation for 2000 ms with time stepping of 0.025 ms
        tSim = 2000
        dt = 0.025
        sim.run(tSim, dt)

.. module:: arbor

.. class:: simulation

    The executable form of a model. A simulation is constructed
    from a recipe, and then used to update and monitor model state.

    Simulations take the following inputs:

        * an :class:`arbor.recipe` that describes the model;
        * an :class:`arbor.domain_decomposition` that describes how the cells in the model are assigned to hardware resources;
        * an :class:`arbor.context` which is used to execute the simulation.

    Simulations provide an interface for executing and interacting with the model:

        * **Advance model state** from one time to another and reset model
          state to its original state before simulation was started.
        * **I/O** interface for sampling simulation state during execution
          (e.g. compartment voltage and current) and spike output.

    **Constructor:**

    .. function:: simulation(recipe, dom_dec, context)

        Initialize the model described by a recipe, with cells and network distributed according to dom_dec, and computation resources described by context.

        .. attribute:: recipe

            See :class:`arbor.recipe`.

        .. attribute:: dom_dec

            See :class:`arbor.domain_decomposition`.

        .. attribute:: context

            See :class:`arbor.context`.

    **Updating Model State:**

    .. function:: reset()

        Reset the state of the simulation to its initial state to rerun the simulation.

    .. function:: run(tfinal, dt)

        Run the simulation from current simulation time to :attr:`tfinal`,
        with maximum time step size :attr:`dt`.

        .. attribute:: tfinal

            Final simulation time (in ms).

        .. attribute:: dt

            Time step size (in ms).

Recording spikes
----------------
In order to analyze the simulation output spikes can be recorded.

**Types**:

.. class:: spike

    .. function:: spike()

        Constructor.

    .. attribute:: source

        The spike source (type: cell_member with :attr:`arbor.cell_member.gid` (default: 0) and :attr:`arbor.cell_member.index` (default: 0)).

    .. attribute:: time

        Spike time (in ms, default: -1 ms).

.. class:: sprec

    .. function:: sprec()

        Constructor.

    .. attribute:: spikes

        Recorded spikes (type: :class:`spike`).

**I/O:**

.. function:: make_spike_recorder(simulation)

       Records all spikes generated over all domains during a simulation (type: :class:`sprec`)

.. container:: example-code

    .. code-block:: python

        import arbor

        # Instatitate the simulation.
        sim = arbor.simulation(recipe, decomp, context)

        # Build the spike recorder
        recorder = arbor.make_spike_recorder(sim)

        # Run the simulation for 2000 ms with time stepping of 0.025 ms
        tSim = 2000
        dt = 0.025
        sim.run(tSim, dt)

        # Get the recorder`s spikes
        spikes = recorder.spikes

        # Print the spikes and according spike time
        for i in range(len(spikes)):
            spike = spikes[i]
            print('  cell %2d at %8.3f ms'%(spike.source.gid, spike.time))

>>> SPIKES:
>>>   cell  0 at    5.375 ms
>>>   cell  1 at   15.700 ms
>>>   cell  2 at   26.025 ms
>>>   cell  3 at   36.350 ms
>>>   cell  4 at   46.675 ms
>>>   cell  5 at   57.000 ms
>>>   cell  6 at   67.325 ms
>>>   cell  7 at   77.650 ms
>>>   cell  8 at   87.975 ms
>>>   cell  9 at   98.300 ms

.. container:: example-code

    .. code-block:: python

        import numpy as np
        import math
        import matplotlib.pyplot as plot

        # Use a raster plot to visualize spiking activity.
        tVec = np.arange(0,tSim,dt)
        SpikeMat_rows = n_cells # number of cells
        SpikeMat_cols = math.floor(tSim/dt)
        SpikeMat = np.zeros((SpikeMat_rows, SpikeMat_cols))

        # save spike trains in matrix:
        # (if spike in cell n at time step k, then SpikeMat[n,k]=1, else 0)
        for i in range(len(spikes)):
            spike = spikes[i]
            tCur = math.floor(spike.time/dt)
            SpikeMat[spike.source.gid][tCur] = 1

        for i in range(SpikeMat_rows):
            for j in range(SpikeMat_cols):
                if(SpikeMat[i,j] == 1):
                    x1 = [i,i+0.5]
                    x2 = [j,j]
                    plot.plot(x2,x1,color = 'black')

        plot.title('Spike raster plot')
        plot.xlabel('Spike time (ms)')
        tick = range(0,SpikeMat_cols+10000,10000)
        label = range(0,tSim+250,250)
        plot.xticks(tick, label)
        plot.ylabel('Neuron (gid)')
        plot.show()


.. figure:: Rasterplot

    Exemplary spike raster plot.
