include(BuildModules.cmake)

# the list of built-in mechanisms to be provided by default
set(mechanisms pas hh expsyn exp2syn test_kin1 test_kinlva test_ca)

set(mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/mod")

# Generate mechanism implementations for host/cpu environment

set(modcc_opt)
set(mech_dir "${CMAKE_CURRENT_SOURCE_DIR}/multicore")
file(MAKE_DIRECTORY "${mech_dir}")
if(ARB_VECTORIZE_TARGET STREQUAL "KNL")
    set(modcc_target "avx512")
elseif(ARB_VECTORIZE_TARGET STREQUAL "AVX")
    set(modcc_opt "-O")
    set(modcc_target "cpu")
elseif(ARB_VECTORIZE_TARGET STREQUAL "AVX2")
    set(modcc_target "avx2")
elseif(ARB_VECTORIZE_TARGET STREQUAL "AVX512")
    set(modcc_target "avx512")
else()
    set(modcc_target "cpu")
endif()
build_modules(
    ${mechanisms}
    SOURCE_DIR "${mod_srcdir}"
    DEST_DIR "${mech_dir}"
    MODCC_FLAGS -t ${modcc_target} ${modcc_opt}
    GENERATES .hpp
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
    GENERATES _impl.cu .hpp _impl.hpp
    TARGET build_all_gpu_mods
)

# Make a library with the implementations of the mechanism kernels

if(ARB_WITH_CUDA)
    # make list of the .cu files that implement the mechanism kernels
    foreach(mech ${mechanisms})
        list(APPEND cuda_mech_sources ${mech_dir}/${mech}_impl.cu)
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
