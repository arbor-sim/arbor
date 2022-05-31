.. _libref:

Utility wrappers and containers
===============================

.. cpp:namespace:: arb::util

.. cpp:class:: unique_any

    A container for a single value of any type that is move constructable.
    Used in the Arbor API where a type of a value passed to or from the API
    is decided at run time.

.. cpp:class:: any_ptr

   Holds a pointer to an arbitrary type, together with the type information.

   .. cpp:function:: template <typename T> T as()

      Retrieve the pointer as type T. If T is ``void *`` or the same
      as the type of the pointer stored in ``any_ptr``, return the held
      value, cast accordingly. Otherwise return ``nullptr``.

   ``any_ptr`` can be used with ``util::any_cast``, so that
   ``util::any_cast<T>(p)`` is equivalent to ``p.as<T>()`` for a value ``p``
   of type ``any_ptr``.

.. cpp:function:: template <typename T> any_cast(...)

    Equivalent to ``std::any_cast`` for ``std::any`` arguments, ``any_cast``
    also performs analogous casting for the :cpp:class:`unique_any` and
    :cpp:class:`any_ptr` utility classes.

    See :ref:`cppcablecell`.
