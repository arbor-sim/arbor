include(${PROJECT_SOURCE_DIR}/mechanisms/BuildModules.cmake)

make_catalogue_lib(
    NAME    ornstein_uhlenbeck
    MOD     ornstein_uhlenbeck
    VERBOSE ${ARB_CAT_VERBOSE})

add_executable(ou EXCLUDE_FROM_ALL ou.cpp)
target_link_libraries(ou PRIVATE catalogue-ornstein_uhlenbeck)
target_link_libraries(ou PRIVATE arbor arborio)
if (ARB_USE_BUNDLED_FMT)
    target_include_directories(ou PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../ext/fmt/include")
    target_compile_definitions(ou PRIVATE FMT_HEADER_ONLY)
else()
    find_package(fmt REQUIRED)
    target_link_libraries(ou PRIVATE fmt::fmt-header-only)
endif()
add_dependencies(examples ou)
