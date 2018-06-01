# Find Julia executable and check presence of packages.
#
# Sets the following variables:
#
#   Julia_FOUND       - True if Julia is found together with requested components.
#   Julia_EXECUTABLE  - Path to Julia executable.
#
#   Julia_Julia_FOUND       - True if Julia is found (regardless of components).
#   Julia_<component>_FOUND - True if the Julia package <component> is found.

include(FindPackageHandleStandardArgs)

if(NOT Julia_FOUND)
    find_program(Julia_EXECUTABLE julia)
    if(Julia_EXECUTABLE)
        set(Julia_Julia_FOUND TRUE)
        foreach(component ${Julia_FIND_COMPONENTS})
            execute_process(
                COMMAND ${Julia_EXECUTABLE} -e "using ${component}"
                RESULT_VARIABLE _result
                OUTPUT_QUIET
                ERROR_QUIET)
            if(_result EQUAL 0)
                set("Julia_${component}_FOUND" TRUE)
            else()
                set("Julia_${component}_FOUND" "Julia_${component}-NOTFOUND")
            endif()
        endforeach()
    else()
        set(Julia_Julia_FOUND Julia_Julia-NOTFOUND)
    endif()

    find_package_handle_standard_args(Julia
        REQUIRED_VARS Julia_EXECUTABLE
        HANDLE_COMPONENTS)
endif()
