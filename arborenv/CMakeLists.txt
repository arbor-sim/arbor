set(arborenv-sources
    arbenvexcept.cpp
    affinity.cpp
    concurrency.cpp
    default_env.cpp
    private_gpu.cpp
    read_envvar.cpp
)

if(ARB_WITH_GPU)
    list(APPEND arborenv-sources gpu_uuid.cpp)
endif()

add_library(arborenv ${arborenv-sources})

add_library(arborenv-public-headers INTERFACE)
target_include_directories(arborenv-public-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# A public arbor dependency is required for the use of arb::mpi_error,
# which in turn requires the arbor libary's singleton for the mpi_error_category
# singleton.

target_link_libraries(arborenv PUBLIC arbor arborenv-public-headers)
target_link_libraries(arborenv PRIVATE arbor-config-defs arborenv-private-deps)

export_visibility(arborenv)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/arborenv/export.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/arborenv)

install(DIRECTORY include/arborenv
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS arborenv-public-headers EXPORT arbor-targets)
install(TARGETS arborenv EXPORT arbor-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
