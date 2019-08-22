# Find the libunwind library
#
#  Unwind_FOUND       - True if libunwind was found
#  Unwind_LIBRARIES   - The libraries needed to use libunwind
#  Unwind_INCLUDE_DIR - Location of unwind.h and libunwind.h
#
# The environment and cmake variables Unwind_ROOT and Unwind_ROOT_DIR
# respectively can be used to help CMake finding the library if it
# is not installed in any of the usual locations.
#
# Registers "Unwind::unwind" as an import library.

if(NOT Unwind_FOUND)
    set(Unwind_SEARCH_DIR ${Unwind_ROOT_DIR} $ENV{Unwind_ROOT})

    find_path(Unwind_INCLUDE_DIR libunwind.h
        HINTS ${Unwind_SEARCH_DIR}
        PATH_SUFFIXES include
    )

    # libunwind requires that we link agains both libunwind.so/a and a
    # a target-specific library libunwind-target.so/a.
    # This code sets the "target" string above in libunwind_arch.
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        set(_libunwind_arch "arm")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
        set(_libunwind_arch "x86_64")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^i.86$")
        set(_libunwind_arch "x86")
    endif()

    find_library(_unwind_library_generic unwind
        HINTS ${Unwind_SEARCH_DIR}
        PATH_SUFFIXES lib64 lib
    )

    find_library(_unwind_library_target unwind-${_libunwind_arch}
        HINTS ${Unwind_SEARCH_DIR}
        PATH_SUFFIXES lib64 lib
    )

    set(Unwind_LIBRARIES ${_unwind_library_generic} ${_unwind_library_target})

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Unwind DEFAULT_MSG Unwind_INCLUDE_DIR Unwind_LIBRARIES)

    mark_as_advanced(Unwind_LIBRARIES Unwind_INCLUDE_DIR)

    if(Unwind_FOUND)
        set(Unwind_INCLUDE_DIRS ${Unwind_INCLUDE_DIR})
        if(NOT TARGET Unwind::unwind)
            add_library(Unwind::unwind UNKNOWN IMPORTED)
            set_target_properties(Unwind::unwind PROPERTIES
                    IMPORTED_LOCATION "${_unwind_library_generic}"
                    INTERFACE_LINK_LIBRARIES "${_unwind_library_target}"
                    INTERFACE_INCLUDE_DIRECTORIES "${Unwind_INCLUDE_DIR}"
            )
        endif()
    endif()

    unset(_unwind_search_dir)
    unset(_unwind_library_generic)
    unset(_unwind_library_target)
    unset(_libunwind_arch)
endif()
