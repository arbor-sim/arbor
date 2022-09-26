.. _modeldomdec:

Domain decomposition
====================

An Arbor simulation requires a :ref:`modelrecipe`, a :ref:`(hardware) context
<modelhardware>`, and a domain decomposition. The Recipe contains the
neuroscientific model, the hardware context describes the computational
resources you are going to execute the simulation on, and the domain
decomposition describes how Arbor will use the hardware. Since the context and
domain decomposition may seem closely related at first, it might be instructive
to see how recipes are used by Arbor:

.. raw:: html
   :file: domdec-diag-1.html

A *domain decomposition* describes the distribution of the model over the
available computational resources. The description partitions the cells in the
model as follows:

* group the cells into cell groups of the same :ref:`kind <modelcellkind>` of
  cell;
* assign each cell group to either a CPU core or GPU on a specific MPI rank.

The number of cells in each cell group depends on different factors, including
the type of the cell, and whether the cell group will run on a CPU core or the
GPU. The domain decomposition is solely responsible for describing the
distribution of cells across cell groups and domains.

The domain decomposition can be built manually by the modeler, or an automatic
load balancer can be used.

We define some terms as used in the context of connectivity

.. glossary::

   connection
      Tuple of ``(source, target, weight, delay)`` describing an
      axon/synapse connection as travelling time (`delay`) and attenuation
      (`weight`) between two sites `source = (gid, threshold_detector)` and `target
      = (gid, synapse)` where `threshold_detector` and `synapse` are string labels.

   cell_group
      List of same-kinded cells that share some information. Must not be split
      across domains.

   domain
      Produced by a `load_balancer`, a list of all `cell_groups`
      located on the same hardware. A `communicator` deals with the full set of
      cells of one `domain`.

   domain_decomposition
      List of domains; distributed across MPI processes.

Load balancers
--------------

A *load balancer* generates the domain decomposition using the model
:ref:`recipe <modelrecipe>` and a description of the available computational
resources on which the model will run described by an :ref:`execution context
<modelcontext>`. Currently Arbor provides one automatic load balancer and a
method for manually listing out the cells per MPI task. More approaches might be
added over time.

API
---

* :ref:`Python <pydomdec>`
* :ref:`C++ <cppdomdec>`

