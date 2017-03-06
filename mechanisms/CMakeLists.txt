include(BuildModules.cmake)

# the list of built-in mechanisms to be provided by default
set(mechanisms pas hh expsyn exp2syn)

set(modcc_opt)
if(NMC_USE_OPTIMIZED_KERNELS) # generate optimized kernels
    set(modcc_opt "-O")
endif()

set(mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/mod")

set(mech_dir "${CMAKE_CURRENT_SOURCE_DIR}/multicore")
file(MAKE_DIRECTORY "${mech_dir}")
if(NMC_VECTORIZE_TARGET STREQUAL "KNL" AND NMC_USE_OPTIMIZED_KERNELS)
    set(modcc_target "cpu")
else()
    set(modcc_target "cpu")
endif()
build_modules(
    ${mechanisms}
    SOURCE_DIR "${mod_srcdir}"
    DEST_DIR "${mech_dir}"
    MODCC_FLAGS -t ${modcc_target} ${modcc_opt}
    TARGET build_all_mods
)

if(NMC_WITH_CUDA)
    set(mech_dir "${CMAKE_CURRENT_SOURCE_DIR}/gpu")
    file(MAKE_DIRECTORY "${mech_dir}")
    build_modules(
        ${mechanisms}
        SOURCE_DIR "${mod_srcdir}"
        DEST_DIR "${mech_dir}"
        MODCC_FLAGS -t gpu ${modcc_opt}
        TARGET build_all_gpu_mods
    )
endif()

