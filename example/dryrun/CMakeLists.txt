add_executable(dryrun EXCLUDE_FROM_ALL dryrun.cpp)
add_dependencies(examples dryrun)

target_link_libraries(dryrun PRIVATE arbor arbor-aux ext-json)
