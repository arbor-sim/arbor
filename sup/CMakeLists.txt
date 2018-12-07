set(sup-sources
    affinity.cpp
    concurrency.cpp
    default_gpu.cpp
    ioutil.cpp
    json_meter.cpp
    path.cpp
    spike_emitter.cpp
)

if(ARB_WITH_MPI)
    list(APPEND sup-sources private_gpu.cpp)
endif()

if(ARB_USE_POSIX_GLOB)
    list(APPEND sup-sources glob_posix.cpp)
else()
    list(APPEND sup-sources glob_basic.cpp)
endif()


add_library(arbor-sup ${sup-sources})

target_compile_options(arbor-sup PRIVATE ${ARB_CXXOPT_ARCH})
target_link_libraries(arbor-sup PUBLIC ext-json arbor)
target_include_directories(arbor-sup PUBLIC include)

if(ARB_WITH_MPI)
    target_compile_definitions(arbor-sup PRIVATE ARB_HAVE_MPI)
endif()
if(ARB_WITH_GPU)
    target_include_directories(arbor-sup PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(arbor-sup PRIVATE ARB_HAVE_GPU)
endif()

set_target_properties(arbor-sup PROPERTIES OUTPUT_NAME arborsup)

