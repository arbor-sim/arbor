add_executable(dryrun dryrun.cpp)

target_link_libraries(dryrun PRIVATE arbor arbor-aux ext-tclap ext-json)
