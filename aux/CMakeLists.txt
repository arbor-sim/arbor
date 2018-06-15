set(aux-sources
    json_meter.cpp)

add_library(arbor-aux ${aux-sources})
target_link_libraries(arbor-aux PUBLIC ext-json)
target_include_directories(arbor-aux INTERFACE .)
