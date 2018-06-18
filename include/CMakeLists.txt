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

# Always make version.hpp-test, but only update version.hpp if it differs.

add_custom_command(
    OUTPUT _always_rebuild
    COMMAND ${CMAKE_COMMAND} -E echo
)

add_custom_command(
    OUTPUT version.hpp-test
    DEPENDS _always_rebuild
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/git-source-id ${PROJECT_VERSION} > version.hpp-test
)

set(version_hpp_path arbor/version.hpp)
add_custom_command(
    OUTPUT ${version_hpp_path}
    DEPENDS version.hpp-test
    COMMAND ${CMAKE_COMMAND} -E copy_if_different version.hpp-test ${version_hpp_path}
)

add_custom_target(generate_version_hpp DEPENDS ${version_hpp_path})
add_dependencies(arbor-public-headers generate_version_hpp)

install(DIRECTORY arbor ${CMAKE_CURRENT_BINARY_DIR}/arbor
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")
