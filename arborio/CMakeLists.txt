set(arborio-sources
    asc_lexer.cpp
    neurolucida.cpp
    swcio.cpp
    cableio.cpp
    cv_policy_parse.cpp
    label_parse.cpp
    neuroml.cpp
    nml_parse_morphology.cpp)

add_library(arborio ${arborio-sources})

if (ARB_USE_BUNDLED_PUGIXML)
    target_include_directories(arborio PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../ext/pugixml/src>)
    target_compile_definitions(arborio PRIVATE PUGIXML_HEADER_ONLY)
else()
    find_package(pugixml REQUIRED)
    target_link_libraries(arborio PUBLIC pugixml::pugixml)
endif()

add_library(arborio-public-headers INTERFACE)
add_library(arborio-private-headers INTERFACE)

target_include_directories(arborio-public-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(arborio-private-headers INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")

target_link_libraries(arborio PUBLIC arbor arborio-public-headers)
list(APPEND arbor_supported_components "neuroml")
set(arbor_supported_components "${arbor_supported_components}" PARENT_SCOPE)

target_link_libraries(arborio PRIVATE arbor-config-defs arborio-private-deps)

export_visibility(arborio)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/arborio/export.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/arborio)

install(DIRECTORY include/arborio
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS arborio-public-headers EXPORT arbor-targets)
install(TARGETS arborio EXPORT arbor-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
