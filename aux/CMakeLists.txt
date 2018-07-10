set(aux-sources

    glob.cpp
    ioutil.cpp
    json_meter.cpp
    path.cpp
    spike_emitter.cpp
)

add_library(arbor-aux ${aux-sources})
target_link_libraries(arbor-aux PUBLIC ext-json arbor)
target_include_directories(arbor-aux PUBLIC include)
set_target_properties(arbor-aux PROPERTIES OUTPUT_NAME arboraux)
