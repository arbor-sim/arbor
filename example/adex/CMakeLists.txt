add_executable(adex EXCLUDE_FROM_ALL adex.cpp)
add_dependencies(examples adex)
target_link_libraries(adex PRIVATE arbor arborio arborenv arbor-sup ext-tinyopt)
