make_catalogue_standalone(
    NAME ou
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}"
    MOD ornstein_uhlenbeck
    CXX
    CXX_FLAGS_TARGET ${ARB_CXX_FLAGS_TARGET_FULL}
    VERBOSE ON)

add_executable(ou EXCLUDE_FROM_ALL ou.cpp)
add_dependencies(ou ou-catalogue)
target_compile_options(ou PRIVATE ${ARB_CXX_FLAGS_TARGET_FULL})
target_include_directories(ou PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/ou")

target_link_libraries(ou PRIVATE arbor arborio ou-catalogue)
if (ARB_USE_BUNDLED_FMT)
    target_include_directories(ou PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../ext/fmt/include")
    target_compile_definitions(ou PRIVATE FMT_HEADER_ONLY)
else()
    find_package(fmt REQUIRED)
    target_link_libraries(ou PRIVATE fmt::fmt-header-only)
endif()

add_dependencies(examples ou)
