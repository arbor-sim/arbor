include(BuildModules.cmake)

# the list of built-in mechanisms to be provided by default
set(mechanisms pas hh expsyn exp2syn test_kin1 test_kinlva test_ca nax kdrmt kamt)

set(mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/mod")

# Generate mechanism implementations for host/cpu environment

set(mech_dir "mechanisms/generated")
set(abs_mech_dir "${CMAKE_SOURCE_DIR}/${mech_dir}")
file(MAKE_DIRECTORY "${abs_mech_dir}")

if(NOT ARB_VECTORIZE STREQUAL "none")
    set(modcc_simd "-s ${ARB_VECTORIZE}")
endif()

build_modules(
    ${mechanisms}
    SOURCE_DIR "${mod_srcdir}"
    DEST_DIR "${abs_mech_dir}"
#    MODCC_FLAGS -t cpu -t gpu ${modcc_simd}
#    GENERATES .hpp _cpu.cpp _gpu_impl.cu _gpu.hpp _gpu_impl.hpp
    MODCC_FLAGS -t cpu ${modcc_simd}
    GENERATES .hpp _cpu.cpp
    TARGET build_all_mods
)

# Generate source for default mechanism catalogue

set(catsrc ${CMAKE_CURRENT_BINARY_DIR}/default_catalogue.cpp)
set(default_catalogue_options -I ${mech_dir} -B multicore -o ${catsrc})
add_custom_command(
    OUTPUT ${catsrc}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_default_catalogue ${default_catalogue_options} ${mechanisms}
    DEPENDS build_all_mods generate_default_catalogue
)

# Make a library with the implementations of the mechanism kernels

foreach(mech ${mechanisms})
    list(APPEND cpu_mech_sources ${abs_mech_dir}/${mech}_cpu.cpp)
endforeach()

add_library(arbormech ${cpu_mech_sources} ${catsrc})
target_compile_options(arbormech PRIVATE ${CXXOPT_ARCH})

if (ARB_AUTO_RUN_MODCC_ON_CHANGES)
    add_dependencies(arbormech build_all_mods)
endif()

# Until we merge our myriad static libraries, we prepend mech libs, and also append CPU mech lib
# which holds the generated default catalogue code.

list(INSERT ARB_LIBRARIES 0 arbormech)
list(APPEND ARB_LIBRARIES arbormech)

if(ARB_WITH_CUDA)
    # make list of the .cu files that implement the mechanism kernels
    foreach(mech ${mechanisms})
        list(APPEND cuda_mech_sources ${abs_mech_dir}/${mech}_gpu_impl.cu)
    endforeach()

    # compile the .cu files into a library
    cuda_add_library(arbormechcu ${cuda_mech_sources})

    # force recompilation on changes to modcc or the underlying .mod files
    if (ARB_AUTO_RUN_MODCC_ON_CHANGES)
        add_dependencies(arbormechcu build_all_gpu_mods)
    endif()

    list(INSERT ARB_LIBRARIES 0 arbormechcu)
endif()

set(ARB_LIBRARIES "${ARB_LIBRARIES}" PARENT_SCOPE)
