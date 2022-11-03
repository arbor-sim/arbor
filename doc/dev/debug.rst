.. _dev-debug:

Debugging
=========

Backtraces
----------

When building Arbor you can enable backtraces in the CMake configure step by
setting ``ARB_BACKTRACE=ON``. Beware aware that this requires the ``Boost``
libraries to be installed on your system. This will cause the following
additions to Arbor's behaviour

1. Failed assertions via ``asb_assert`` will print the corresponding stacktrace.
2. All exceptions deriving from ``arbor_exception`` and ``arbor_internal_error``
   will have stacktraces attached in the ``where`` field.
3. Python exceptions derived from these types will add that same stacktrace
   information to their message.

Alternatively, you can obtain the same information using a debugger like GDB or
LLDB.

.. note::

   Since Arbor often uses a buffer of instructions on how to construct a
   particular object instead of perform the action right away, errors occur not
   always at the location you might expect.

   Consider this (adapted from ``network_ring.py``)

   .. code-block:: python

      class Recipe(arb.recipe):
        # [...]
        def connections_on(self, gid):
          return [arbor.connection((src, "detector"), "syn-NOT", w, d)]

        def cell_description(self, gid):
          # [...]
          decor = (arbor.decor()
              .place('"synapse_site"', arbor.synapse("expsyn"), "syn"))
          return arbor.cable_cell(tree, decor, labels)

      rec = Recipe()            # "Ok"
      sim = arb.simulation(rec) # ERROR here
