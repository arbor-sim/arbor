set(sup-sources
    affinity.cpp
    concurrency.cpp
    glob.cpp
    default_gpu.cpp
    ioutil.cpp
    json_meter.cpp
    path.cpp
    spike_emitter.cpp
)

if(ARB_WITH_GPU)
    list(APPEND sup-sources gpu_uuid.cpp)
endif()

if(ARB_WITH_MPI)
    list(APPEND sup-sources private_gpu.cpp)
endif()

add_library(arbor-sup ${sup-sources})

# Compile sup library with the same optimization flags as libarbor.
target_compile_options(arbor-sup PRIVATE ${ARB_CXXOPT_ARCH})

# The sup library uses both the json library and libarbor
target_link_libraries(arbor-sup PUBLIC ext-json arbor)

target_include_directories(arbor-sup PUBLIC include)

if(ARB_WITH_MPI)
    target_compile_definitions(arbor-sup PRIVATE ARB_HAVE_MPI)
endif()
if(ARB_WITH_GPU)
    target_compile_definitions(arbor-sup PRIVATE ARB_HAVE_GPU)
    # So that cpp and hpp files can use cuda.h and cuda_runtime.h
    target_include_directories(arbor-sup PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # The sup lib needs to use the CUDA NVML library for CUDA 9
    #if (${CUDA_VERSION_MAJOR} LESS 10)
        find_library(nvml_lib_path
                     NAMES nvidia-ml
                     PATHS ${CUDA_TOOLKIT_ROOT_DIR}
                     PATH_SUFFIXES lib64/stubs)
        if (NOT nvml_lib_path)
            message(FATAL_ERROR "Unable to find CUDA NVML library: libnvida-ml.so")
        endif()
        target_link_libraries(arbor-sup PUBLIC ${nvml_lib_path})
    #endif()
endif()

set_target_properties(arbor-sup PROPERTIES OUTPUT_NAME arborsup)

