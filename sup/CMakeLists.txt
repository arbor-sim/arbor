set(sup-sources
    ioutil.cpp
    json_meter.cpp
)

add_library(arbor-sup ${sup-sources})

if (ARB_USE_BUNDLED_FMT)
    target_include_directories(arbor-sup
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../ext/fmt/include>)

    target_compile_definitions(arbor-sup PRIVATE FMT_HEADER_ONLY)
else()
    target_include_directories(arbor-sup
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
    find_package(fmt REQUIRED)
    target_link_libraries(arbor-sup PRIVATE fmt::fmt-header-only)
endif()

# Compile sup library with the same optimization flags as libarbor.
target_compile_options(arbor-sup PRIVATE ${ARB_CXX_FLAGS_TARGET_FULL})

# The sup library uses both the json library and libarbor
target_link_libraries(arbor-sup PUBLIC ${json_library_name} arbor)

#target_include_directories(arbor-sup PUBLIC include)
target_include_directories(arbor-sup
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

set_target_properties(arbor-sup PROPERTIES OUTPUT_NAME arborsup)

export_visibility(arbor-sup)
