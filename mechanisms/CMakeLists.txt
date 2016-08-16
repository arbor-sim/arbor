set(mechanisms pas hh expsyn exp2syn)

set(modcc "${CMAKE_BINARY_DIR}/external/bin/modcc")

set(modcc_flags "-t cpu")
if(TARGET_VECTORIZE)
    set(modcc_flags ${modcc_flags}  -O)
endif()

foreach(mech ${mechanisms})
    set(mod "${CMAKE_CURRENT_SOURCE_DIR}/mod/${mech}.mod")
    set(hpp "${CMAKE_CURRENT_SOURCE_DIR}/${mech}.hpp")
    add_custom_command(OUTPUT "${hpp}"
                       DEPENDS modparser "${mod}"
                       WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       COMMAND ${modcc} ${modcc_flags} ${mod} -o ${hpp})
    set_source_files_properties("${hpp}" PROPERTIES GENERATED TRUE)
    list(APPEND all_mod_hpps "${hpp}")
endforeach()

# Fake target to always trigger .mod -> .hpp dependencies because wtf CMake
add_custom_target(build_all_mods DEPENDS ${all_mod_hpps})

