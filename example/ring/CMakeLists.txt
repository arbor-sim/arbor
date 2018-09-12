add_executable(ring ring.cpp)

target_link_libraries(ring PRIVATE arbor arbor-anc ext-tclap ext-json)
