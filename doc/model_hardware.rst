.. _modelhardware:

Hardware
========

*Local resources* are locally available computational resources, specifically the number of hardware threads and the number of GPUs.

An *allocation* enumerates the computational resources to be used for a simulation, typically a subset of the resources available on a physical hardware node.

.. Note::

   New users can find using contexts a little verbose.
   The design is very deliberate, to allow fine-grained control over which
   computational resources an Arbor simulation should use.
   As a result Arbor is much easier to integrate into workflows that
   run multiple applications or libraries on the same node, because
   Arbor has a direct API for using on node resources (threads and GPU)
   and distributed resources (MPI) that have been partitioned between
   applications/libraries.


Execution Context
-----------------

An *execution context* contains the local thread pool, and optionally the GPU state and MPI communicator, if available. Users of the library configure contexts, which are passed to Arbor methods and types.

See :ref:`pyhardware` for documentation of the Python interface and :ref:`cpphardware` for the C++ interface for managing hardware resources.
