set(arbornml-sources
    arbornml.cpp
    nmlexcept.cpp
    parse_morphology.cpp
    with_xml.cpp
    xmlwrap.cpp
)

find_package(LibXml2 REQUIRED)

add_library(arbornml ${arbornml-sources})

add_library(arbornml-public-headers INTERFACE)
target_include_directories(arbornml-public-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(arbornml PUBLIC arbor arbornml-public-headers LibXml2::LibXml2)
target_link_libraries(arbornml PRIVATE arbor-config-defs arbornml-private-deps)

install(DIRECTORY include/arbornml
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS arbornml-public-headers EXPORT arbor-targets)
install(TARGETS arbornml EXPORT arbor-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

list(APPEND arbor_export_dependencies "LibXml2")
set(arbor_export_dependencies "${arbor_export_dependencies}" PARENT_SCOPE)
list(APPEND arbor_supported_components "neuroml")
set(arbor_supported_components "${arbor_supported_components}" PARENT_SCOPE)
