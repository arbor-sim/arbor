add_executable(neighborhood EXCLUDE_FROM_ALL neighborhood.cpp)
add_dependencies(examples neighborhood)

target_link_libraries(neighborhood PRIVATE arbor arborenv arbor-sup ext-tinyopt)
