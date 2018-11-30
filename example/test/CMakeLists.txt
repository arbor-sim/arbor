add_executable(test EXCLUDE_FROM_ALL
    test.cpp
)
add_dependencies(examples test)

target_link_libraries(test PRIVATE arbor arbor-sup ext-tclap ext-json)
