.. _modelsimulation:

Simulations
===========
A simulation is the executable form of a model and is used to interact with and monitor the model state. In the simulation the neuron model is initiated and the spike exchange and the integration for each cell group are scheduled.

From recipe to simulation
-------------------------

To build a simulation the following are needed:

    * A recipe that describes the cells and connections in the model.
    * A context used to execute the simulation.

The workflow to build a simulation is to first generate a domain decomposition that describes the distribution of the model over the local and distributed hardware resources (see :ref:`modeldomdec`), then build the simulation from the recipe, the domain decomposition and the execution context. Optionally experimental inputs  that can change between model runs, such as external spike trains, can be injected.

The recipe describes the model, the domain decomposition describes how the cells in the model are assigned to hardware resources and the context is used to execute the simulation.

Simulation execution and interaction
------------------------------------

Simulations provide an interface for executing and interacting with the model:

    * The simulation is executed/ *run* by advancing the model state from the current simulation time to another with maximum time step size.
    * The model state can be *reset* to its initial state before the simulation was started.
    * *Sampling* of the simulation state can be performed during execution with samplers and probes (e.g. compartment voltage and current) and spike output with the total number of spikes generated since either construction or reset.

The documentation for Arbor's Python simulation interface can be found in :ref:`pysimulation`.
See :ref:`cppsimulation` for documentation of the C++ simulation API.
