.. _cppoverview:

C++
=========

The C++ API is the recommended interface through which advanced users and HPC
developers can access Arbor.

Arbor makes a distinction between the **description** of a model, and the
**execution** of a model.

A :cpp:type:`arb::recipe` describes a model, and a :cpp:type:`arb::simulation` is an executable instantiation of a model.

.. toctree::
   :caption: C++ API:
   :maxdepth: 2

   recipe
   cell
   interconnectivity
   remote
   event_generators
   hardware
   domdec
   mechanisms
   simulation
   profiler
   cable_cell
   lif_cell
   spike_source_cell
   labels
