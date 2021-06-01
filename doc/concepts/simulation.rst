.. _modelsimulation:

Simulations
===========
A simulation is the executable form of a model and is used to interact with and monitor the model state.
In the simulation the neuron model is initiated and the spike exchange and the integration for each cell
group are scheduled.

From recipe to simulation
-------------------------

To build a simulation the following are needed:

* A :ref:`recipe <modelrecipe>` that describes the cells and connections in the model.
* A :ref:`domain decomposition <modeldomdec>` that describes the distribution of the
  model over the local and distributed :ref:`hardware resources <modelhardware>`.
* An :ref:`execution context <modelcontext>` used to execute the simulation.

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
