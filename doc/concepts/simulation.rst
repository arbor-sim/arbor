.. _modelsimulation:

Simulations
===========
A simulation is the executable form of a model and is used to interact with and monitor the model state.
In the simulation the neuron model is initiated and the spike exchange and the integration for each cell
group are scheduled.

From recipe to simulation
-------------------------

To build a simulation the following is needed:

* A :ref:`recipe <modelrecipe>` that describes the cells and connections in the model.
* A :ref:`domain decomposition <modeldomdec>` that describes the distribution of
  the model over the local and distributed :ref:`hardware resources
  <modelhardware>`. If not given, a default algorithm will be used which assigns
  cells to groups one to one; each group is assigned to a thread from the context.
* An :ref:`execution context <modelcontext>` used to execute the simulation. If
  not given, the default context will be used, which allocates one thread, one
  process (MPI task), and no GPU.

Configuring data extraction
---------------------------

Two kinds of data extraction can be set up:

# certain state variables can be :ref:`sampled <probesample>` by attaching a
  sampler to a probe.
# spikes can be recorded by either a callback (C++) or a preset recording model
  (Python), see the API docs linked below.

Simulation execution and interaction
------------------------------------

Simulations provide an interface for executing and interacting with the model:

* The simulation is executed/*run* by advancing the model state from the current simulation time to another
  with maximum time step size.
* The model state can be *reset* to its initial state before the simulation was started.
* *Sampling* of the simulation state can be performed during execution with samplers and probes
  and spike output with the total number of spikes generated since either construction or reset.

API
---

* :ref:`Python <pysimulation>`
* :ref:`C++ <cppsimulation>`
