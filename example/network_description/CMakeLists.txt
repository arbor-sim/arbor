add_executable(network_description EXCLUDE_FROM_ALL network_description.cpp)
add_dependencies(examples network_description)

target_link_libraries(network_description PRIVATE arbor arborio arborenv arbor-sup ${json_library_name})
