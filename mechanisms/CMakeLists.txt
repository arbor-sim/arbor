include(BuildModules.cmake)

# the list of built-in mechanisms to be provided by default
set(mechanisms pas hh expsyn exp2syn test_kin1 test_kinlva)

set(modcc_opt)

set(mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/mod")

set(mech_dir "${CMAKE_CURRENT_SOURCE_DIR}/multicore")
file(MAKE_DIRECTORY "${mech_dir}")
if(NMC_VECTORIZE_TARGET STREQUAL "KNL")
    set(modcc_target "avx512")
elseif(NMC_VECTORIZE_TARGET STREQUAL "AVX")
    set(modcc_opt "-O")
    set(modcc_target "cpu")
elseif(NMC_VECTORIZE_TARGET STREQUAL "AVX2")
    set(modcc_target "avx2")
elseif(NMC_VECTORIZE_TARGET STREQUAL "AVX512")
    set(modcc_target "avx512")
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

set(mech_dir "${CMAKE_CURRENT_SOURCE_DIR}/gpu")
file(MAKE_DIRECTORY "${mech_dir}")
build_modules(
    ${mechanisms}
    SOURCE_DIR "${mod_srcdir}"
    DEST_DIR "${mech_dir}"
    MODCC_FLAGS -t gpu ${modcc_opt}
    TARGET build_all_gpu_mods
)

