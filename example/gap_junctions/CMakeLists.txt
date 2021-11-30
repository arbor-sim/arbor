add_executable(gap_junctions EXCLUDE_FROM_ALL gap_junctions.cpp)
add_dependencies(examples gap_junctions)

target_link_libraries(gap_junctions PRIVATE arbor arborio arborenv arbor-sup ${json_library_name})
