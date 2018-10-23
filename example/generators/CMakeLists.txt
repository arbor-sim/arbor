add_executable(event-gen EXCLUDE_FROM_ALL event_gen.cpp)
add_dependencies(examples event-gen)

target_link_libraries(event-gen PRIVATE arbor arbor-sup ext-json)
