add_executable(ring ring.cpp)

target_link_libraries(ring PRIVATE arbor arbor-aux ext-json)

# TODO: resolve public headers
target_link_libraries(ring PRIVATE arbor-private-headers)
