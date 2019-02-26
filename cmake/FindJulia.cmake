# Find Julia executable and check presence of packages.
#
# Sets the following variables:
#
#   Julia_FOUND       - True if Julia is found together with requested components.
#   Julia_EXECUTABLE  - Path to Julia executable, or "Julia_EXECUTABLE-NOTFOUND" if not found.
#
#   Julia_<component>_FOUND - True if the Julia package <component> is found.
#
# The Julia_EXECUTABLE and Julia_<component>_FOUND variables are cached.

include(FindPackageHandleStandardArgs)

if(NOT Julia_FOUND)
    find_program(Julia_EXECUTABLE julia)
    if(Julia_EXECUTABLE)
        if(Julia_FIND_COMPONENTS)
            message(STATUS "Checking for Julia components: ${Julia_FIND_COMPONENTS}")
        endif()
        foreach(component ${Julia_FIND_COMPONENTS})
            set(_found_var "Julia_${component}_FOUND")
            if(NOT ${_found_var})
                execute_process(
                    COMMAND ${Julia_EXECUTABLE} -e "using ${component}"
                    RESULT_VARIABLE _result
                    OUTPUT_QUIET
                    ERROR_QUIET)
                if(_result EQUAL 0)
                    set(${_found_var} TRUE CACHE BOOL "Found Julia component $component" FORCE)
                else()
                    set(${_found_var} FALSE CACHE BOOL "Found Julia component $component" FORCE)
                endif()
            endif()
        endforeach()
    endif()

    find_package_handle_standard_args(Julia
        REQUIRED_VARS Julia_EXECUTABLE
        HANDLE_COMPONENTS)
endif()
