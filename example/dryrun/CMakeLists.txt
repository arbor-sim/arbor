add_executable(dryrun EXCLUDE_FROM_ALL dryrun.cpp)
add_dependencies(examples dryrun)

target_link_libraries(dryrun PRIVATE arbor arborio arborenv arbor-sup ${json_library_name})
