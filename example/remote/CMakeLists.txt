add_executable(remote EXCLUDE_FROM_ALL remote.cpp)
add_dependencies(examples remote)

target_link_libraries(remote PRIVATE arbor arborio arborenv arbor-sup ${json_library_name})
