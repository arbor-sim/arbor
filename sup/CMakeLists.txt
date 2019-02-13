set(sup-sources
    glob_basic.cpp
    ioutil.cpp
    json_meter.cpp
    path.cpp
)

if(ARB_USE_POSIX_GLOB)
    list(APPEND sup-sources glob_posix.cpp)
else()
    list(APPEND sup-sources glob_basic_wrap.cpp)
endif()


add_library(arbor-sup ${sup-sources})

# Compile sup library with the same optimization flags as libarbor.
target_compile_options(arbor-sup PRIVATE ${ARB_CXXOPT_ARCH})

# The sup library uses both the json library and libarbor
target_link_libraries(arbor-sup PUBLIC ext-json arbor)

target_include_directories(arbor-sup PUBLIC include)

set_target_properties(arbor-sup PROPERTIES OUTPUT_NAME arborsup)

