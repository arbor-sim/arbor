set(sup-sources
    affinity.cpp
    concurrency.cpp
    glob.cpp
    gpu.cpp
    ioutil.cpp
    json_meter.cpp
    path.cpp
    spike_emitter.cpp
)

add_library(arbor-sup ${sup-sources})
target_compile_options(arbor-sup PRIVATE ${ARB_CXXOPT_ARCH})
target_link_libraries(arbor-sup PUBLIC ext-json arbor)
target_include_directories(arbor-sup PUBLIC include)
set_target_properties(arbor-sup PROPERTIES OUTPUT_NAME arborsup)
