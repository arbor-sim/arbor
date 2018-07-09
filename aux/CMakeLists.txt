set(aux-sources json_meter.cpp glob.cpp path.cpp)

add_library(arbor-aux ${aux-sources})
target_link_libraries(arbor-aux PUBLIC ext-json arbor)
target_include_directories(arbor-aux PUBLIC include)
set_target_properties(arbor-aux PROPERTIES OUTPUT_NAME arboraux)
