# Ancillary library for tests and examples.

set(anc-sources

    glob.cpp
    ioutil.cpp
    json_meter.cpp
    path.cpp
    spike_emitter.cpp
)

add_library(arbor-anc ${anc-sources})
target_compile_options(arbor-anc PRIVATE ${ARB_CXXOPT_ARCH})
target_link_libraries(arbor-anc PUBLIC ext-json arbor)
target_include_directories(arbor-anc PUBLIC include)
set_target_properties(arbor-anc PROPERTIES OUTPUT_NAME arboranc)
