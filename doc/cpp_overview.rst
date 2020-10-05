.. _cppoverview:

Overview
=========

The C++ API is the recommended interface through which advanced users and HPC
developers can access Arbor.

Arbor makes a distinction between the **description** of a model, and the
**execution** of a model.

A :cpp:type:`arb::recipe` describes a model, and a :cpp:type:`arb::simulation` is an executable instantiation of a model.
