add_executable(single-cell EXCLUDE_FROM_ALL single.cpp)
add_dependencies(examples single-cell)
target_link_libraries(single-cell PRIVATE arbor arborenv arbor-sup ext-tinyopt)
