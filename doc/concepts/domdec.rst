.. _modeldomdec:

Domain decomposition
====================

A *domain decomposition* describes the distribution of the model over the available computational resources.
The description partitions the cells in the model as follows:

* group the cells into cell groups of the same :ref:`kind <modelcellkind>` of cell;
* assign each cell group to either a CPU core or GPU on a specific MPI rank.

The number of cells in each cell group depends on different factors, including the type of the cell, and whether the
cell group will run on a CPU core or the GPU. The domain decomposition is solely responsible for describing the distribution
of cells across cell groups and domains.

The domain decomposition can be built manually by the modeler, or an automatic load balancer can be used.


Load balancers
--------------

A *load balancer* generates the domain decomposition using the model :ref:`recipe <modelrecipe>` and a description of
the available computational resources on which the model will run described by an :ref:`execution context <modelcontext>`.
Currently Arbor provides one load balancer and more will be added over time.

API
---

* :ref:`Python <pydomdec>`
* :ref:`C++ <cppdomdec>`

