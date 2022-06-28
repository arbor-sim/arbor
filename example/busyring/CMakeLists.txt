add_executable(busyring EXCLUDE_FROM_ALL ring.cpp)
add_dependencies(examples busyring)

target_link_libraries(busyring PRIVATE arbor arborio arborenv arbor-sup ${json_library_name})
