add_executable(gj_network EXCLUDE_FROM_ALL
    gj_network.cpp
)
add_dependencies(examples gj_network)

target_link_libraries(gj_network PRIVATE arbor arbor-sup ext-tclap ext-json)
