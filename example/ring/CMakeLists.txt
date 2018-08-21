add_executable(ring ring.cpp)

target_link_libraries(ring PRIVATE arbor arbor-aux ext-tclap ext-json)
