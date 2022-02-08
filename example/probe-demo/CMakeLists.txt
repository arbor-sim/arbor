add_executable(probe-demo EXCLUDE_FROM_ALL probe-demo.cpp)
add_dependencies(examples probe-demo)
target_link_libraries(probe-demo PRIVATE arbor ext-tinyopt)
set_target_properties(probe-demo PROPERTIES CXX_VISIBILITY_PRESET hidden)
