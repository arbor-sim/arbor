include(BuildModules.cmake)

# the list of built-in mechanisms to be provided by default
set(mechanisms pas hh expsyn exp2syn test_kin1 test_kinlva test_ca nax kdrmt kamt)

set(mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/mod")

# Generate mechanism implementations for host/cpu environment

set(mech_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${mech_dir}")

if(ARB_VECTORIZE)
    set(modcc_simd "-s")
endif()

build_modules(
    ${mechanisms}
    SOURCE_DIR "${mod_srcdir}"
    DEST_DIR "${mech_dir}"
    MODCC_FLAGS -t cpu -t gpu ${modcc_simd}
    GENERATES .hpp _cpu.cpp _gpu.cpp _gpu.cu
    TARGET build_all_mods
)

# Generate source for default mechanism catalogue

set(catsrc ${CMAKE_CURRENT_BINARY_DIR}/default_catalogue.cpp)
set(default_catalogue_options -I ${mech_dir} -o ${catsrc} -B multicore)
if(ARB_WITH_CUDA)
    list(APPEND default_catalogue_options -B gpu)
endif()

add_custom_command(
    OUTPUT ${catsrc}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_default_catalogue ${default_catalogue_options} ${mechanisms}
    DEPENDS build_all_mods generate_default_catalogue
)

# Make libraries with the implementations of the mechanism kernels.

foreach(mech ${mechanisms})
    list(APPEND cpu_mech_sources ${mech_dir}/${mech}_cpu.cpp)
endforeach()

add_library(arbormech ${cpu_mech_sources} ${catsrc})
target_compile_options(arbormech PRIVATE ${CXXOPT_ARCH})
set(mech_libs arbormech)

if (ARB_AUTO_RUN_MODCC_ON_CHANGES)
    add_dependencies(arbormech build_all_mods)
endif()

if(ARB_WITH_CUDA)
    foreach(mech ${mechanisms})
        list(APPEND cuda_mech_sources "${mech_dir}/${mech}_gpu.cpp" "${mech_dir}/${mech}_gpu.cu")
    endforeach()

    cuda_add_library(arbormechcu ${cuda_mech_sources})
    set(mech_libs arbormech arbormechcu)

    # force recompilation on changes to modcc or the underlying .mod files
    if (ARB_AUTO_RUN_MODCC_ON_CHANGES)
        add_dependencies(arbormechcu build_all_mods)
    endif()

endif()

# Until we merge our myriad static libraries, we prepend mech libs and also append mech libs
# to capture generated default catalogue code interdependencies with arbor lib.

list(INSERT ARB_LIBRARIES 0 ${mech_libs})
list(APPEND ARB_LIBRARIES ${mech_libs})

set(ARB_LIBRARIES "${ARB_LIBRARIES}" PARENT_SCOPE)
