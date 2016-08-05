set(mechanisms pas hh expsyn exp2syn)

set(modcc "${CMAKE_BINARY_DIR}/external/bin/modcc")

foreach(mech ${mechanisms})
    set(mod "${CMAKE_CURRENT_SOURCE_DIR}/mod/${mech}.mod")
    set(hpp "${CMAKE_CURRENT_SOURCE_DIR}/${mech}.hpp")
    add_custom_command(OUTPUT "${hpp}"
                       DEPENDS modparser "${mod}"
                       WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       COMMAND "${modcc}" -t cpu -O -o "${hpp}" "${mod}")
    set_source_files_properties("${hpp}" PROPERTIES GENERATED TRUE)
    list(APPEND all_mod_hpps "${hpp}")
endforeach()

# Fake target to always trigger .mod -> .hpp dependencies because wtf CMake
add_custom_target(build_all_mods DEPENDS ${all_mod_hpps})

