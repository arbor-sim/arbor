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

