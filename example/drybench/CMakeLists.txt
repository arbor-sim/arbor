add_executable(drybench EXCLUDE_FROM_ALL drybench.cpp)
add_dependencies(examples drybench)

target_link_libraries(drybench PRIVATE arbor arborenv arbor-sup ${json_library_name})
