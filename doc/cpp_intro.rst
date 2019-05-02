.. _cppoverview:

Overview
=========

The C++ API for is the main interface through which application developers will
access Arbor, though it is designed to be usable for power users to
implement models.

Arbor makes a distinction between the **description** of a model, and the
**execution** of a model.

A :cpp:type:`arb::recipe` describes a model, and a :cpp:type:`arb::simulation` is an executable instantiation of a model.
