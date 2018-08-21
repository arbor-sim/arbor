add_executable(bench bench.cpp recipe.cpp parameters.cpp)

target_link_libraries(bench PRIVATE arbor arbor-aux ext-tclap ext-json)
