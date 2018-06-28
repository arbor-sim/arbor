add_executable(threadbench bench.cpp)

target_link_libraries(threadbench PRIVATE arbor arbor-aux ext-tclap ext-json)

# TODO: resolve public headers
target_link_libraries(threadbench PRIVATE arbor-private-headers)
target_link_libraries(threadbench PRIVATE arbor-private-deps)



