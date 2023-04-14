# Distributed as part of Arbor.

#[=======================================================================[.rst:
Findhwloc
-------

Finds the hwloc library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``hwloc::hwloc``
  The hwloc library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``hwloc_FOUND``
  True if the system has the hwloc library.
``hwloc_VERSION``
  The version of the hwloc library which was found.
``hwloc_INCLUDE_DIRS``
  Include directories needed to use hwloc.
``hwloc_LIBRARIES``
  Libraries needed to link to hwloc.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``hwloc_INCLUDE_DIR``
  The directory containing ``hwloc.h``.
``hwloc_LIBRARY``
  The path to the hwloc library.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_package(PkgConfig)

pkg_check_modules(PC_hwloc QUIET hwloc)

find_path(hwloc_INCLUDE_DIR
          NAMES hwloc.h
          PATHS ${PC_hwloc_INCLUDE_DIRS}
          PATH_SUFFIXES hwloc)

find_library(hwloc_LIBRARY
             NAMES hwloc
             PATHS ${PC_hwloc_LIBRARY_DIRS})

set(hwloc_VERSION ${PC_hwloc_VERSION})

find_package_handle_standard_args(hwloc
                                  FOUND_VAR hwloc_FOUND
                                  REQUIRED_VARS hwloc_LIBRARY hwloc_INCLUDE_DIR
                                  VERSION_VAR hwloc_VERSION)

if(hwloc_FOUND AND NOT TARGET hwloc::hwloc)
  add_library(hwloc::hwloc UNKNOWN IMPORTED)
  set_target_properties(hwloc::hwloc PROPERTIES
                        IMPORTED_LOCATION "${hwloc_LIBRARY}"
                        INTERFACE_COMPILE_OPTIONS "${PC_hwloc_CFLAGS_OTHER}"
                        INTERFACE_INCLUDE_DIRECTORIES "${hwloc_INCLUDE_DIR}")
endif()

mark_as_advanced(hwloc_INCLUDE_DIR hwloc_LIBRARY)
