.. _modeldomdec:

Domain Decomposition
====================

A *domain decomposition* describes the distribution of the model over the available computational resources. The description partitions the cells in the model as follows:

    * group the cells into cell groups of the same kind of cell;
    * assign each cell group to either a CPU core or GPU on a specific MPI rank.

The number of cells in each cell group depends on different factors, including the type of the cell, and whether the cell group will run on a CPU core or the GPU. The domain decomposition is soley responsible for describing the distribution of cells across cell groups and domains.


Load Balancers
--------------

A *load balancer* generates the domain decomposition using the
model recipe and a description of the available computational resources on which the model will run described by an execution context.
Currently Arbor provides one load balancer and more will be added over time.


Hardware
--------

*Local resources* are locally available computational resources, specifically the number of hardware threads and the number of GPUs.

An *allocation* enumerates the computational resources to be used for a simulation, typically a subset of the resources available on a physical hardware node.

Execution Context
-----------------

An *execution context* contains the local thread pool, and optionally the GPU state and MPI communicator, if available. Users of the library configure contexts, which are passed to Arbor methods and types.

Detailed documentations can be found in :ref:`cppdomdec` for C++ and in :ref:`pydomdec` for python.
