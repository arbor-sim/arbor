add_executable(ring EXCLUDE_FROM_ALL ring.cpp)
add_dependencies(examples ring)

target_link_libraries(ring PRIVATE arbor arbor-sup ext-tclap ext-json)
