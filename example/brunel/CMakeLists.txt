add_executable(brunel EXCLUDE_FROM_ALL brunel.cpp)
add_dependencies(examples brunel)

target_link_libraries(brunel PRIVATE arbor arborenv arbor-sup ext-tinyopt)
#set_target_properties(brunel PROPERTIES CXX_VISIBILITY_PRESET hidden)
