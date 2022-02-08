add_executable(bench EXCLUDE_FROM_ALL bench.cpp)
add_dependencies(examples bench)

target_link_libraries(bench PRIVATE arbor arborenv arbor-sup ${json_library_name})
set_target_properties(bench PROPERTIES CXX_VISIBILITY_PRESET hidden)
