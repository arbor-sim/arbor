# Find the libunwind library
#
#  UNWIND_FOUND       - True if libunwind was found
#  UNWIND_LIBRARIES   - The libraries needed to use libunwind
#  UNWIND_INCLUDE_DIR - Location of unwind.h and libunwind.h
#
# The environment and cmake variables UNWIND_ROOT and UNWIND_ROOT_DIR
# respectively can be used to help CMake finding the library if it
# is not installed in any of the usual locations.

if(NOT UNWIND_FOUND)
    set(UNWIND_SEARCH_DIR ${UNWIND_ROOT_DIR} $ENV{UNWIND_ROOT})

    find_path(UNWIND_INCLUDE_DIR libunwind.h
        HINTS ${UNWIND_SEARCH_DIR}
        PATH_SUFFIXES include
    )

    # libunwind requires that we link agains both libunwind.so/a and a
    # a target-specific library libunwind-target.so/a.
    # This code sets the "target" string above in libunwind_arch.
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        set(libunwind_arch "arm")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
        set(libunwind_arch "x86_64")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^i.86$")
        set(libunwind_arch "x86")
    endif()

    find_library(unwind_library_generic unwind
        HINTS ${UNWIND_SEARCH_DIR}
        PATH_SUFFIXES lib64 lib
    )

    find_library(unwind_library_target unwind-${libunwind_arch}
        HINTS ${UNWIND_SEARCH_DIR}
        PATH_SUFFIXES lib64 lib
    )

    set(UNWIND_LIBRARIES ${unwind_library_generic} ${unwind_library_target})

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(UNWIND DEFAULT_MSG UNWIND_INCLUDE_DIR UNWIND_LIBRARIES)

    mark_as_advanced(UNWIND_LIBRARIES UNWIND_INCLUDE_DIR)

    unset(unwind_search_dir)
    unset(unwind_library_generic)
    unset(unwind_library_target)
    unset(libunwind_arch)
endif()
