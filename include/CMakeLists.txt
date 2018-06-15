file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/arbor)
if(ARB_WITH_ASSERTIONS)
configure_file(arbor/assert_macro.hpp.enabled arbor/assert_macro.hpp COPYONLY)
else()
configure_file(arbor/assert_macro.hpp.disabled arbor/assert_macro.hpp COPYONLY)
endif()

add_library(arbor-public-headers INTERFACE)
target_include_directories(arbor-public-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(DIRECTORY arbor ${CMAKE_CURRENT_BINARY_DIR}/arbor
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")
