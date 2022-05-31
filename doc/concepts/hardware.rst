.. _modelhardware:

Hardware context
================

An Arbor simulation requires a :ref:`modelrecipe`, a (hardware) context, and a :ref:`modeldomdec`. The Recipe contains the neuroscientific model, the hardware context describes the computational resources you are going to execute the simulation on, and the domain decomposition describes how Arbor will use the hardware. Since the context and domain decomposition may seem closely related at first, it might be instructive to see how recipes are used by Arbor: 

.. raw:: html
   :file: domdec-diag-1.html

*Local resources* are locally available computational resources, specifically the number of hardware threads and the number of GPUs.

An *allocation* enumerates the computational resources to be used for a simulation, typically a subset of the resources available on a physical hardware node.

New users can find using contexts a little verbose.
The design is very deliberate, to allow fine-grained control over which
computational resources an Arbor simulation should use.
As a result Arbor is much easier to integrate into workflows that
run multiple applications or libraries on the same node, because
Arbor has a direct API for using on node resources (threads and GPU)
and distributed resources (MPI) that have been partitioned between
applications/libraries.


.. _modelcontext:

Execution context
-----------------

An *execution context* contains the local thread pool, and optionally the GPU state and MPI communicator, if available. Users of the library configure contexts, which are passed to Arbor methods and types.

API
---

* :ref:`Python <pyhardware>`
* :ref:`C++ <cpphardware>`
