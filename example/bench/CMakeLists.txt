add_executable(bench EXCLUDE_FROM_ALL bench.cpp recipe.cpp parameters.cpp)
add_dependencies(examples bench)

target_link_libraries(bench PRIVATE arbor arbor-sup ext-tclap ext-json)
