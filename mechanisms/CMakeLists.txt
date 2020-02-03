include(BuildModules.cmake)

# The list of library-provided mechanisms used to populate the default catalogue:
set(mechanisms pas hh expsyn exp2syn nernst test_kin1 test_kinlva test_ca nax kdrmt kamt)

set(mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/mod")

# Generate mechanism implementations.

set(mech_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${mech_dir}")

set(external_modcc)
if(ARB_WITH_EXTERNAL_MODCC)
    set(external_modcc MODCC ${modcc})
endif()

build_modules(
    ${mechanisms}
    SOURCE_DIR "${mod_srcdir}"
    DEST_DIR "${mech_dir}"
    ${external_modcc}
    MODCC_FLAGS -t cpu -t gpu ${ARB_MODCC_FLAGS}
    GENERATES .hpp _cpu.cpp _gpu.cpp _gpu.cu
    TARGET build_all_mods
)

# Generate source for default mechanism catalogue.

set(catsrc ${CMAKE_CURRENT_BINARY_DIR}/default_catalogue.cpp)
set(default_catalogue_options -A arbor -I ${mech_dir} -o ${catsrc} -B multicore)
if(ARB_WITH_CUDA)
    list(APPEND default_catalogue_options -B gpu)
endif()

add_custom_command(
    OUTPUT ${catsrc}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_default_catalogue ${default_catalogue_options} ${mechanisms}
    DEPENDS generate_default_catalogue
)

add_custom_target(default_catalogue_cpp_target DEPENDS ${catsrc})
add_dependencies(build_all_mods default_catalogue_cpp_target)

# Make libraries with the implementations of the mechanism kernels.

set(mech_sources ${catsrc})
foreach(mech ${mechanisms})
    list(APPEND mech_sources ${mech_dir}/${mech}_cpu.cpp)
    if(ARB_WITH_CUDA)
        list(APPEND mech_sources ${mech_dir}/${mech}_gpu.cpp)
        list(APPEND mech_sources ${mech_dir}/${mech}_gpu.cu)
    endif()
endforeach()

set(arbor_mechanism_sources ${mech_sources} PARENT_SCOPE)
