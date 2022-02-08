add_executable(drybench EXCLUDE_FROM_ALL drybench.cpp)
add_dependencies(examples drybench)

target_link_libraries(drybench PRIVATE arbor arborenv arbor-sup ext-json)
set_target_properties(drybench PROPERTIES CXX_VISIBILITY_PRESET hidden)
