add_executable(generators EXCLUDE_FROM_ALL generators.cpp)
add_dependencies(examples generators)

target_link_libraries(event-gen PRIVATE arbor arbor-sup ext-json)
