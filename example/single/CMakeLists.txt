add_executable(single-cell EXCLUDE_FROM_ALL single.cpp)
add_dependencies(examples single-cell)
target_link_libraries(single-cell PRIVATE arbor arborio arborenv arbor-sup ext-tinyopt)
set_target_properties(single-cell PROPERTIES CXX_VISIBILITY_PRESET hidden)
