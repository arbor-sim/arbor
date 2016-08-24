
# Only build modcc if it has not already been installed.
# This is useful if cross compiling for KNL, when it is not desirable to compile
# modcc with the same flags that are used for the KNL target.
find_program(MODCC_BIN modcc)
set(modcc "${MODCC_BIN}")
set(use_external_modcc OFF BOOL)

# the modcc executable was not found, so build our own copy
if(MODCC_BIN STREQUAL "MODCC_BIN-NOTFOUND")
    include(ExternalProject)
    externalproject_add(modparser
        PREFIX ${CMAKE_BINARY_DIR}/external
        CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external"
                   "-DCMAKE_CXX_FLAGS=${SAVED_CXX_FLAGS}"
                   "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        BINARY_DIR "${CMAKE_BINARY_DIR}/external/modparser"
        STAMP_DIR  "${CMAKE_BINARY_DIR}/external/"
        TMP_DIR    "${CMAKE_BINARY_DIR}/external/tmp"
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/modparser"
    )
    # Set up environment to use the version of modcc that is compiled
    # as the ExternalProject above.
    set(use_external_modcc ON)
    set(modcc "${CMAKE_BINARY_DIR}/external/bin/modcc")
endif()

# the list of built-in mechanisms to be provided by default
set(mechanisms pas hh expsyn exp2syn)

# set the flags for the modcc compiler that converts NMODL
# files to C++/CUDA source.
set(modcc_flags "-t cpu")
if(USE_OPTIMIZED_KERNELS) # generate optimized kernels
    set(modcc_flags ${modcc_flags} -O)
endif()

# generate source for each mechanism
foreach(mech ${mechanisms})
    set(mod "${CMAKE_CURRENT_SOURCE_DIR}/mod/${mech}.mod")
    set(hpp "${CMAKE_CURRENT_SOURCE_DIR}/${mech}.hpp")
    if(use_external_modcc)
        add_custom_command(
           OUTPUT "${hpp}"
           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
           COMMAND ${modcc} ${modcc_flags} ${mod} -o ${hpp}
       )
    else()
        add_custom_command(
            OUTPUT "${hpp}"
            DEPENDS modparser "${mod}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMAND ${modcc} ${modcc_flags} ${mod} -o ${hpp}
        )
    endif()
    set_source_files_properties("${hpp}" PROPERTIES GENERATED TRUE)
    list(APPEND all_mod_hpps "${hpp}")
endforeach()

# Fake target to always trigger .mod -> .hpp dependencies because wtf CMake
add_custom_target(build_all_mods DEPENDS ${all_mod_hpps})

