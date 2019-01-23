envset(arborenv-sources
    affinity.cpp
    concurrency.cpp
    default_gpu.cpp
    private_gpu.cpp
)

if(ARB_WITH_GPU)
    list(APPEND arborenv-sources gpu_uuid.cpp)
endif()

add_library(arborenv ${arborenv-sources})

add_library(arborenv-public-headers INTERFACE)
target_include_directories(arborenv-public-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# A public arbor dependency is required for the use of arb::mpi_error,
# which in turn requires the arbor libary's singleton for the mpi_error_category
# singleton.

target_link_libraries(arborenv PUBLIC arbor arborenv-public-headers)
target_link_libraries(arborenv PRIVATE arbor-config-defs arborenv-private-deps)

if(ARB_WITH_GPU)
    # So that cpp and hpp files can use cuda.h and cuda_runtime.h
    target_include_directories(arborenv PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # The arborenv lib needs to use the CUDA NVML library for CUDA 9
    if (${CUDA_VERSION_MAJOR} LESS 10)
        # TODO: Wrap this in a FindNVML.cmake script?
        find_library(nvml_lib_path
                     NAMES nvidia-ml
                     PATHS ${CUDA_TOOLKIT_ROOT_DIR}
                     PATH_SUFFIXES lib64/stubs)

        if (NOT nvml_lib_path)
            message(FATAL_ERROR "Unable to find CUDA NVML library: libnvida-ml.so")
        endif()

        target_link_libraries(arborenv PUBLIC ${nvml_lib_path})
    endif()
endif()

set_target_properties(arborenv PROPERTIES OUTPUT_NAME arborenv)

