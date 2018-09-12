add_executable(risotto risotto.cpp)

target_link_libraries(risotto PRIVATE arbor arbor-anc ext-tclap ext-json)
