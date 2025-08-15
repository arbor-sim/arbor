include(${PROJECT_SOURCE_DIR}/mechanisms/BuildModules.cmake)

make_catalogue_lib(
    NAME    ornstein_uhlenbeck
    MOD     ornstein_uhlenbeck
    VERBOSE ${ARB_CAT_VERBOSE})

add_executable(ou EXCLUDE_FROM_ALL ou.cpp)
target_link_libraries(ou PRIVATE catalogue-ornstein_uhlenbeck arbor arborio fmt::fmt-header-only)
add_dependencies(examples ou)
