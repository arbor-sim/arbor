add_executable(plasticity EXCLUDE_FROM_ALL plasticity.cpp)
add_dependencies(examples plasticity)

target_link_libraries(plasticity PRIVATE arbor arborio arborenv)
