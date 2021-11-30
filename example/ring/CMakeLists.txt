add_executable(ring EXCLUDE_FROM_ALL ring.cpp)
add_dependencies(examples ring)

target_link_libraries(ring PRIVATE arbor arborio arborenv arbor-sup ${json_library_name})
