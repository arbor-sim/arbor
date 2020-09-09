.. _cppoverview:

Overview
=========

The C++ API for is the main interface through which application developers will
access Arbor, though it is designed to be usable for power users to
implement models.

Arbor makes a distinction between the **description** of a model, and the
**execution** of a model.

A :cpp:type:`arb::recipe` describes a model, and a :cpp:type:`arb::simulation` is an executable instantiation of a model.

Utility Wrappers and Containers
--------------------------------

.. cpp:namespace:: arb::util


.. cpp:class:: template <typename T> optional

    A wrapper around a contained value of type :cpp:type:`T`, that may or may not be set.
    A faithful copy of the C++17 ``std::optional`` type.
    See the online C++ standard documentation
    `<https://en.cppreference.com/w/cpp/utility/optional>`_
    for more information.

.. cpp:class:: any

    A container for a single value of any type that is copy constructable.
    Used in the Arbor API where a type of a value passed to or from the API
    is decided at run time.

    A faithful copy of the C++17 ``std::any`` type.
    See the online C++ standard documentation
    `<https://en.cppreference.com/w/cpp/utility/any>`_
    for more information.

    The :cpp:any:`arb::util` namespace also implementations of the
    :cpp:any:`any_cast`, :cpp:any:`make_any` and :cpp:any:`bad_any_cast`
    helper functions and types from C++17.

.. cpp:class:: unique_any

   Equivalent to :cpp:class:`util::any`, except that:

      * it can store any type that is move constructable;
      * it is move only, that is it can't be copied.
