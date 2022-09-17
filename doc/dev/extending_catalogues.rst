.. _extending-catalogues:

Adding Built-in Catalogues to Arbor
===================================

There are two ways new mechanisms catalogues can be added to Arbor, statically
or dynamically. If you have a set of mechanisms to use with Arbor, you are in
all likelihood interested in the former.

.. warning::

   If you are coming from NEURON and looking for the equivalent of
   ``nrnivmodl``, please read on :ref:`here <mechanisms_dynamic>`.

   Following the below instructions is for developers rather than end-users.

This requires a copy of the Arbor source tree and the compiler toolchain used to
build Arbor in addition to the installed library. Following these steps will
produce a catalogue of the same level of integration as the built-in catalogues
(*default*, *bbp*, and *allen*). The required steps are as follows

1. Go to the Arbor source tree.
2. Create a new directory under *mechanisms* with the name of your catalogue

   1. Add any ``.mod`` files you wish to integrate.
   2. Add any raw C++ files to be included in the catalogue.

4. Edit *mechanisms/CMakeLists.txt* to add a definition like this (example from
   *default* catalogue)

   .. code-block :: cmake

     make_catalogue(
       NAME default                                                # Name of your catalogue, must match directory under 2.
       MOD exp2syn expsyn expsyn_stdp hh kamt kdrmt nax nernst pas # Space separated list of mechanism names
       CXX                                                         # Optional: list of raw C++ mechanism names
       VERBOSE  ${ARB_CAT_VERBOSE}                                 # Print debug info at configuration time
       ADD_DEPS ON)                                                # Must be ON, make catalogue part of arbor
5. Add a ``global_NAME_catalogue`` function in ``mechcat.hpp``.
6. Bind this function in ``python/mechanisms.cpp`` to ``NAME-catalogue``.

All steps can be directly adapted from the surrounding code.

.. note::

   If you have special requirements, you can write mechanisms in C/C++ directly
   against Arbor's ABI. These need to adhere to the calling convention of the
   ABI. See :ref:`here <abi_raw>` for more.
