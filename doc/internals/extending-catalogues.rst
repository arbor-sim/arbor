Adding Catalogues to Arbor
==========================

There are two ways new mechanisms catalogues can be added to Arbor, statically
or dynamically. None is considered to be part of the stable user-facing API at
the moment, although the dynamic approach is aligned with our eventual goals.

Both require a copy of the Arbor source tree and the compiler toolchain used to
build Arbor in addition to the installed library.

Static Extensions
'''''''''''''''''

This will produce a catalogue of the same level of integration as the built-in catalogues
(*default*, *bbp*, and *allen*). We briefly sketch the required steps here:

1. Go to the Arbor source tree.
1. Create a new directory under *mechanisms*.
1. Add your .mod files.
1. Edit *mechanisms/CMakeLists.txt* to add a definition like this

   .. code-block :: cmake

     make_catalogue(
       NAME <catalogue-name>                             # Name of your catalogue
       SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/<directory>" # Directory name (added above)
       OUTPUT "<output-name>"                            # Variable name to output to
       MECHS <names>)                                    # Space separated list of mechanism
                                                         # names w/o .mod suffix.

1. Add your `output-name` to the `arbor_mechanism_sources` list.
1. Add a `global_NAME_catalogue` function in `mechcat.hpp` and `mechcat.cpp`
1. Bind this function in `python/mechanisms.cpp`.

All steps can be more or less copied from the surrounding code.

Dynamic Extensions
''''''''''''''''''

This will produce a catalogue loadable at runtime by eg calling `load_catalogue` with a
filename in both C++ and Python. The steps are

1. Prepare a directory `dir` containing your NMODL files (.mod suffixes required)
1. Call `build_catalogue.py` from the `scripts` directory like this

   .. code-block :: bash

   build-catalogue.py -C <name> -I <path/to/nmodl> -S <path/to/source/tree> -A <path/to/installed/arbor>

All files with the suffix `.mod` will be baked into a catalogue named `<name>_catalogue.so`.
