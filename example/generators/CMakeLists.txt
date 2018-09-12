add_executable(event-gen event_gen.cpp)

target_link_libraries(event-gen PRIVATE arbor arbor-anc ext-json)
