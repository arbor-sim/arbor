.. _modelsimulation:

Simulations
===========
A simulation is the executable form of a model and is used to interact with and monitor the model state.
In the simulation, the neuron model is initiated and the spike exchange and the integration for each cell
group are scheduled.

From recipe to simulation
-------------------------

To build a simulation, the following is needed:

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

#. certain state variables can be :ref:`sampled <probesample>` by attaching a sampler to a probe.
#. spikes can be recorded by either a callback (C++) or a preset recording model (Python), see the API docs linked below.

Simulation execution and interaction
------------------------------------

Simulations provide an interface for executing and interacting with the model:

* The simulation is executed/*run* by advancing the model state from the current simulation time to another
  with maximum time step size.
* The model state can be *reset* to its initial state before the simulation was started.
* *Sampling* of the simulation state can be performed during execution with samplers and probes
  and spike output with the total number of spikes generated since either construction or reset.

.. _modelspikeexchange:

Spike exchange
--------------

Spikes travel between cells on different MPI ranks. By default, all ranks exchange their
spikes after every epoch, where an epoch is half the minimum delay of the whole network.
This exchange involves all ranks, so its cost grows with their number.

Many networks are spatially structured, with cells placed in space and connected mostly
to nearby cells, with delays that grow with distance. If the domain decomposition keeps
nearby cells on the same or neighbouring ranks, most connections between ranks are
between neighbouring ranks, and connections between distant ranks have longer delays. For
such networks, the *neighbourhood spike exchange* can make communication much cheaper. It
splits the ranks that a rank receives spikes from into levels:

* the rank itself, which is handled without communication,
* one level per *neighbourhood*, a set of ranks with which spikes are exchanged with a
  sparse MPI neighbourhood collective, involving only those ranks,
* all remaining *far* ranks, with which spikes are exchanged globally.

Spikes are buffered per level, and each level is only exchanged as often as the minimum
delay of the connections from it allows; spikes still arrive in time for all their events.
If, for example, the delays between distant ranks are ten times the minimum delay, the
global exchange runs only every few epochs. All buffers are flushed at the end of each run
of the simulation. Without such structure, for example in a randomly connected network
with uniform delays, the neighbourhood exchange brings no benefit.

The method and its performance are described in `Scalable Spike Transmission in
Large-Scale Brain Network Simulations <https://doi.org/10.1109/ipdps65963.2026.00034>`_
(IPDPS 2026). How spikes are exchanged internally, by default and with neighbourhoods, is
described in the :ref:`developer documentation <communication>`.

The levels can be given by the user or chosen by arbor:

Given levels
    Each rank lists the ranks it receives spikes from, grouped into neighbourhoods,
    innermost first. All ranks must list the same number of neighbourhoods, and no rank
    may appear twice in the same neighbourhood. A rank may appear in several
    neighbourhoods; it then belongs to the first of them. This allows nested levels, where
    each level repeats the ranks of the previous one. The calling rank itself is ignored.
    All ranks not listed are far ranks. Even without any neighbourhoods, buffering the far
    spikes saves global exchanges if the delays between ranks exceed the minimum delay.

Automatic levels
    Every rank puts each rank it receives from into a class by the exchange period that the
    minimum delay of their connections allows: periods of 1, 2–3, 4–7, 8–15 epochs, and so
    on. With :math:`K` the slowest class over all ranks, classes :math:`0` to :math:`K - 1`
    become neighbourhoods and class :math:`K` the far level. Each level is then exchanged
    at most twice as often as its own delays require. If all delays are equal, all ranks
    end up as far ranks, which is the default exchange. The levels are chosen anew whenever
    the connections change.

The simulation reports the levels in use on each rank: the ranks in each level, its
exchange period in epochs, and the minimum delay of its connections from which the period
follows.

In a dry run, give the levels of rank 0. Since every rank holds a copy of rank 0's tile,
shifted by its rank, the levels of rank :math:`r` are taken to be those of rank 0, shifted
by :math:`r`.

API
---

* :ref:`Python <pysimulation>`
* :ref:`C++ <cppsimulation>`
