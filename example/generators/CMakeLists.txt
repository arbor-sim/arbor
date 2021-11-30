add_executable(generators EXCLUDE_FROM_ALL generators.cpp)
add_dependencies(examples generators)

target_link_libraries(generators PRIVATE arbor arborio arbor-sup ${json_library_name})
