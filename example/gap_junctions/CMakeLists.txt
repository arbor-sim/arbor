add_executable(gap_junctions gap_junctions.cpp)

target_link_libraries(gap_junctions PRIVATE arbor arbor-aux ext-json)
