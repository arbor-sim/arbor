.. _extending-catalogues:

Adding Catalogues to Arbor
==========================

There are two ways new mechanisms catalogues can be added to Arbor, statically
or dynamically. None is considered to be part of the stable user-facing API at
the moment, although the dynamic approach is aligned with our eventual goals.

Both require a copy of the Arbor source tree and the compiler toolchain used to
build Arbor in addition to the installed library.

Static Extensions
'''''''''''''''''

This will produce a catalogue of the same level of integration as the built-in
catalogues (*default*, *bbp*, and *allen*). The required steps are as follows

1. Go to the Arbor source tree.
2. Create a new directory under *mechanisms*.
3. Add your .mod files.
4. Edit *mechanisms/CMakeLists.txt* to add a definition like this

   .. code-block :: cmake

     make_catalogue(
       NAME <catalogue-name>                             # Name of your catalogue
       SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/<directory>" # Directory name (added above)
       OUTPUT "<output-name>"                            # Variable name to output to
       CXX_FLAGS_TARGET "<compiler flags>"               # Target-specific flags for C++ compiler
       MECHS <names>)                                    # Space separated list of mechanism
                                                         # names w/o .mod suffix.

5. Add your `output-name` to the `arbor_mechanism_sources` list.
6. Add a `global_NAME_catalogue` function in `mechcat.hpp` and `mechcat.cpp`
7. Bind this function in `python/mechanisms.cpp`.

All steps can be more or less copied from the surrounding code.

Dynamic Extensions
''''''''''''''''''

This will produce a catalogue loadable at runtime by calling `load_catalogue`
with a filename in both C++ and Python. The steps are

1. Prepare a directory containing your NMODL files (.mod suffixes required)
2. Call `build_catalogue` from the `scripts` directory

   .. code-block :: bash

     build-catalogue <name> <path/to/nmodl>

All files with the suffix `.mod` located in `<path/to/nmodl>` will be baked into
a catalogue named `lib<name>-catalogue.so` and placed into your current working
directory. Note that these files are platform-specific and should only be used
on the combination of OS, compiler, arbor, and machine they were built with.

See the demonstration in `python/example/dynamic-catalogue.py` for an example.
