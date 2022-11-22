include(${PROJECT_SOURCE_DIR}/mechanisms/BuildModules.cmake)

make_catalogue(
    NAME    ornstein_uhlenbeck
    MOD     ornstein_uhlenbeck
    VERBOSE ${ARB_CAT_VERBOSE}
    ADD_DEPS OFF)

add_executable(ou EXCLUDE_FROM_ALL ou.cpp ${catalogue-ornstein_uhlenbeck-mechanisms})
add_dependencies(ou catalogue-ornstein_uhlenbeck-target)

if(ARB_WITH_CUDA_CLANG OR ARB_WITH_HIP_CLANG)
    set_source_files_properties(${catalogue-ornstein_uhlenbeck-meachanisms} PROPERTIES LANGUAGE CXX)
endif()
target_include_directories(ou PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/ornstein_uhlenbeck")

target_link_libraries(ou PRIVATE arbor-private-deps) # compiler flags, GPU options etc
target_link_libraries(ou PRIVATE arbor arborio)
if (ARB_USE_BUNDLED_FMT)
    target_include_directories(ou PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../ext/fmt/include")
    target_compile_definitions(ou PRIVATE FMT_HEADER_ONLY)
else()
    find_package(fmt REQUIRED)
    target_link_libraries(ou PRIVATE fmt::fmt-header-only)
endif()
add_dependencies(examples ou)
