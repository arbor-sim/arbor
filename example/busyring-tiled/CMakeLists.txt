add_executable(tiled-busyring EXCLUDE_FROM_ALL ring.cpp)
add_dependencies(examples tiled-busyring)

target_link_libraries(tiled-busyring PRIVATE arbor arborio arborenv arbor-sup ${json_library_name})
