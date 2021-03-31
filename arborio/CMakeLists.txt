set(arborio-sources
    asc_lexer.cpp
    neurolucida.cpp
    swcio.cpp
    cableio.cpp
)
if(ARB_WITH_NEUROML)
    list(APPEND arborio-sources
            neuroml.cpp
            nml_parse_morphology.cpp
            xml.cpp
            xmlwrap.cpp
        )
    find_package(LibXml2 REQUIRED)
endif()


add_library(arborio ${arborio-sources})

add_library(arborio-public-headers INTERFACE)
add_library(arborio-private-headers INTERFACE)

target_include_directories(arborio-public-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(arborio-private-headers INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")

if(ARB_WITH_NEUROML)
    target_link_libraries(arborio PUBLIC arbor arborio-public-headers LibXml2::LibXml2)
    list(APPEND arbor_export_dependencies "LibXml2")
    set(arbor_export_dependencies "${arbor_export_dependencies}" PARENT_SCOPE)
    list(APPEND arbor_supported_components "neuroml")
    set(arbor_supported_components "${arbor_supported_components}" PARENT_SCOPE)
else ()
    target_link_libraries(arborio PUBLIC arbor arborio-public-headers)
endif()

target_link_libraries(arborio PRIVATE arbor-config-defs arborio-private-deps)

install(DIRECTORY include/arborio
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS arborio-public-headers EXPORT arbor-targets)
install(TARGETS arborio EXPORT arbor-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
