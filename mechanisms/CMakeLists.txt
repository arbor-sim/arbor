include(BuildModules.cmake)

# Define catalogues
make_catalogue(
  NAME bbp
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bbp"
  MOD CaDynamics_E2 Ca_HVA Ca_LVAst Ih Im K_Pst K_Tst Nap_Et2 NaTa_t NaTs2_t SK_E2 SKv3_1
  CXX
  PREFIX "${PROJECT_SOURCE_DIR}/mechanisms"
  CXX_FLAGS_TARGET "${ARB_CXX_FLAGS_TARGET_FULL}"
  STANDALONE FALSE
  VERBOSE ${ARB_CAT_VERBOSE})

make_catalogue(
  NAME allen
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/allen"
  MOD CaDynamics Ca_HVA Ca_LVA Ih Im Im_v2 K_P K_T Kd Kv2like Kv3_1 NaTa NaTs NaV Nap SK
  CXX
  PREFIX "${PROJECT_SOURCE_DIR}/mechanisms"
  CXX_FLAGS_TARGET "${ARB_CXX_FLAGS_TARGET_FULL}"
  STANDALONE FALSE
  VERBOSE ${ARB_CAT_VERBOSE})

make_catalogue(
  NAME default
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/default"
  MOD exp2syn expsyn expsyn_curr expsyn_stdp hh kamt kdrmt nax nernst pas gj decay inject
  CXX
  PREFIX "${PROJECT_SOURCE_DIR}/mechanisms"
  CXX_FLAGS_TARGET "${ARB_CXX_FLAGS_TARGET_FULL}"
  STANDALONE FALSE
  VERBOSE ${ARB_CAT_VERBOSE})

set(arbor-builtin-mechanisms ${catalogue-default-mechs} ${catalogue-allen-mechs} ${catalogue-bbp-mechs} PARENT_SCOPE)

add_dependencies(arbor-public-deps catalogue-default-target)
add_dependencies(arbor-public-deps catalogue-allen-target)
add_dependencies(arbor-public-deps catalogue-bbp-target)

if(ARB_WITH_CUDA_CLANG OR ARB_WITH_HIP_CLANG)
    set_source_files_properties(${arbor-builtin-mechs} PROPERTIES LANGUAGE CXX)
endif()
