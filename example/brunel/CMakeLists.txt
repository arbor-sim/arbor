add_executable(brunel EXCLUDE_FROM_ALL brunel.cpp)
add_dependencies(examples brunel)

target_link_libraries(brunel PRIVATE arbor arborenv arbor-sup ext-tinyopt)
