add_executable(voltage-clamp EXCLUDE_FROM_ALL v-clamp.cpp)
add_dependencies(examples voltage-clamp)
target_link_libraries(voltage-clamp PRIVATE arbor arborio arborenv arbor-sup ext-tinyopt)
