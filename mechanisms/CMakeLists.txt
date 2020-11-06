include(BuildModules.cmake)

set(external_modcc)
if(ARB_WITH_EXTERNAL_MODCC)
    set(external_modcc MODCC ${modcc})
endif()

add_custom_target(build_all_mods)

set(mech_sources "")

# BBP
set(bbp_mechanisms CaDynamics_E2 Ca_HVA Ca_LVAst Ih Im K_Pst K_Tst Nap_Et2 NaTa_t NaTs2_t SK_E2 SKv3_1)
set(bbp_mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/bbp")
set(mech_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/bbp")
file(MAKE_DIRECTORY "${mech_dir}")

build_modules(
        ${bbp_mechanisms}
        SOURCE_DIR "${bbp_mod_srcdir}"
        DEST_DIR "${mech_dir}"
        ${external_modcc}
        MODCC_FLAGS -t cpu -t gpu ${ARB_MODCC_FLAGS} -N arb::bbp_catalogue
        GENERATES .hpp _cpu.cpp _gpu.cpp _gpu.cu
        TARGET build_catalogue_bbp_mods)

set(bbp_catalogue_source ${CMAKE_CURRENT_BINARY_DIR}/bbp_catalogue.cpp)
set(bbp_catalogue_options -A arbor -I ${mech_dir} -o ${bbp_catalogue_source} -B multicore -C bbp -N arb::bbp_catalogue)
if(ARB_WITH_GPU)
    list(APPEND bbp_catalogue_options -B gpu)
endif()

add_custom_command(
        OUTPUT ${bbp_catalogue_source}
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_catalogue ${bbp_catalogue_options} ${bbp_mechanisms}
        DEPENDS generate_catalogue
)

add_custom_target(bbp_catalogue_cpp_target DEPENDS ${bbp_catalogue_source})
add_dependencies(build_catalogue_bbp_mods bbp_catalogue_cpp_target)
add_dependencies(build_all_mods build_catalogue_bbp_mods)

list(APPEND mech_sources ${bbp_catalogue_source})
foreach(mech ${bbp_mechanisms})
    list(APPEND mech_sources ${mech_dir}/${mech}_cpu.cpp)
    if(ARB_WITH_GPU)
        list(APPEND mech_sources ${mech_dir}/${mech}_gpu.cpp)
        list(APPEND mech_sources ${mech_dir}/${mech}_gpu.cu)
    endif()
endforeach()

# ALLEN
set(allen_mechanisms CaDynamics Ca_HVA Ca_LVA Ih Im Im_v2 K_P K_T Kd Kv2like Kv3_1 NaTa NaTs NaV Nap SK)
set(allen_mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/allen")
set(mech_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/allen")
file(MAKE_DIRECTORY "${mech_dir}")

build_modules(
    ${allen_mechanisms}
    SOURCE_DIR "${allen_mod_srcdir}"
    DEST_DIR "${mech_dir}"
    ${external_modcc}
    MODCC_FLAGS -t cpu -t gpu ${ARB_MODCC_FLAGS} -N arb::allen_catalogue
    GENERATES .hpp _cpu.cpp _gpu.cpp _gpu.cu
    TARGET build_catalogue_allen_mods)

set(allen_catalogue_source ${CMAKE_CURRENT_BINARY_DIR}/allen_catalogue.cpp)
set(allen_catalogue_options -A arbor -I ${mech_dir} -o ${allen_catalogue_source} -B multicore -C allen -N arb::allen_catalogue)
if(ARB_WITH_GPU)
    list(APPEND allen_catalogue_options -B gpu)
endif()

add_custom_command(
    OUTPUT ${allen_catalogue_source}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_catalogue ${allen_catalogue_options} ${allen_mechanisms}
    DEPENDS generate_catalogue
)

add_custom_target(allen_catalogue_cpp_target DEPENDS ${allen_catalogue_source})
add_dependencies(build_catalogue_allen_mods allen_catalogue_cpp_target)
add_dependencies(build_all_mods build_catalogue_allen_mods)

list(APPEND mech_sources ${allen_catalogue_source})
foreach(mech ${allen_mechanisms})
    list(APPEND mech_sources ${mech_dir}/${mech}_cpu.cpp)
    if(ARB_WITH_GPU)
        list(APPEND mech_sources ${mech_dir}/${mech}_gpu.cpp)
        list(APPEND mech_sources ${mech_dir}/${mech}_gpu.cu)
    endif()
endforeach()

# DEFAULT
set(default_mechanisms exp2syn expsyn hh kamt kdrmt nax nernst pas)
set(default_mod_srcdir "${CMAKE_CURRENT_SOURCE_DIR}/default")
set(mech_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/default")
file(MAKE_DIRECTORY "${mech_dir}")

build_modules(
    ${default_mechanisms}
    SOURCE_DIR "${default_mod_srcdir}"
    DEST_DIR "${mech_dir}"
    ${external_modcc}
    MODCC_FLAGS -t cpu -t gpu ${ARB_MODCC_FLAGS} -N arb::default_catalogue
    GENERATES .hpp _cpu.cpp _gpu.cpp _gpu.cu
    TARGET build_catalogue_default_mods)

set(default_catalogue_source ${CMAKE_CURRENT_BINARY_DIR}/default_catalogue.cpp)
set(default_catalogue_options -A arbor -I ${mech_dir} -o ${default_catalogue_source} -B multicore -C default -N arb::default_catalogue)
if(ARB_WITH_GPU)
    list(APPEND default_catalogue_options -B gpu)
endif()

add_custom_command(
    OUTPUT ${default_catalogue_source}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_catalogue ${default_catalogue_options} ${default_mechanisms}
    DEPENDS generate_catalogue)

add_custom_target(default_catalogue_cpp_target DEPENDS ${default_catalogue_source})
add_dependencies(build_catalogue_default_mods default_catalogue_cpp_target)
add_dependencies(build_all_mods build_catalogue_default_mods)

list(APPEND mech_sources ${default_catalogue_source})
foreach(mech ${default_mechanisms})
    list(APPEND mech_sources ${mech_dir}/${mech}_cpu.cpp)
    if(ARB_WITH_GPU)
        list(APPEND mech_sources ${mech_dir}/${mech}_gpu.cpp)
        list(APPEND mech_sources ${mech_dir}/${mech}_gpu.cu)
    endif()
endforeach()

# 
set(arbor_mechanism_sources ${mech_sources} PARENT_SCOPE)
if(ARB_WITH_CUDA_CLANG OR ARB_WITH_HIP_CLANG)
    set_source_files_properties(${arbor_mechanism_sources} PROPERTIES LANGUAGE CXX)
endif()
