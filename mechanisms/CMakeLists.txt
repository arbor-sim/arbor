include(BuildModules.cmake)

# the list of built-in mechanisms to be provided by default
set(mechanisms pas hh expsyn exp2syn test_kin1 test_kinlva test_ca)

set(mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/mod")

# Generate mechanism implementations for host/cpu environment

set(mech_dir "${CMAKE_CURRENT_SOURCE_DIR}/multicore")
file(MAKE_DIRECTORY "${mech_dir}")

if(ARB_VECTORIZE_TARGET STREQUAL "none")
    set(modcc_simd "")
elseif(ARB_VECTORIZE_TARGET STREQUAL "KNL")
    set(modcc_simd "-s avx512")
elseif(ARB_VECTORIZE_TARGET STREQUAL "AVX512")
    set(modcc_simd "-s avx512")
elseif(ARB_VECTORIZE_TARGET STREQUAL "AVX2")
    set(modcc_simd "-s avx2")
else()
    message(SEND_ERROR "Unrecognized architecture for ARB_VECTORIZE_TARGET")
    set(modcc_simd "")
endif()
build_modules(
    ${mechanisms}
    SOURCE_DIR "${mod_srcdir}"
    DEST_DIR "${mech_dir}"
    MODCC_FLAGS -t cpu ${modcc_simd}
    GENERATES _cpu.hpp
    TARGET build_all_mods
)

# Generate mechanism implementations for gpu

set(mech_dir "${CMAKE_CURRENT_SOURCE_DIR}/gpu")
file(MAKE_DIRECTORY "${mech_dir}")
build_modules(
    ${mechanisms}
    SOURCE_DIR "${mod_srcdir}"
    DEST_DIR "${mech_dir}"
    MODCC_FLAGS -t gpu
    GENERATES _gpu_impl.cu _gpu.hpp _gpu_impl.hpp
    TARGET build_all_gpu_mods
)

# Make a library with the implementations of the mechanism kernels

if(ARB_WITH_CUDA)
    # make list of the .cu files that implement the mechanism kernels
    foreach(mech ${mechanisms})
        list(APPEND cuda_mech_sources ${mech_dir}/${mech}_gpu_impl.cu)
    endforeach()

    # compile the .cu files into a library
    cuda_add_library(arbormechcu ${cuda_mech_sources})

    # force recompilation on changes to modcc or the underlying .mod files
    if (ARB_AUTO_RUN_MODCC_ON_CHANGES)
        add_dependencies(arbormechcu build_all_gpu_mods)
    endif()

    list(APPEND ARB_LIBRARIES arbormechcu)
    set(ARB_LIBRARIES "${ARB_LIBRARIES}" PARENT_SCOPE)
endif()
