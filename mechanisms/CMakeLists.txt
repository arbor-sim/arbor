include(BuildModules.cmake)

set(arbor-builtin-mechanisms)

# Define catalogues
make_catalogue(
  NAME bbp
  MOD CaDynamics_E2 Ca_HVA Ca_LVAst Ih Im K_Pst K_Tst Nap_Et2 NaTa_t NaTs2_t SK_E2 SKv3_1
  VERBOSE ${ARB_CAT_VERBOSE}
  ADD_DEPS ON)

make_catalogue(
  NAME allen
  MOD CaDynamics Ca_HVA Ca_LVA Ih Im Im_v2 K_P K_T Kd Kv2like Kv3_1 NaTa NaTs NaV Nap SK
  VERBOSE ${ARB_CAT_VERBOSE}
  ADD_DEPS ON)

make_catalogue(
  NAME default
  MOD exp2syn expsyn expsyn_curr expsyn_stdp hh kamt kdrmt nax nernst pas gj decay inject v_clamp v_limit
  VERBOSE ${ARB_CAT_VERBOSE}
  ADD_DEPS ON)

make_catalogue(
  NAME stochastic
  MOD ou_input calcium_based_synapse
  VERBOSE ${ARB_CAT_VERBOSE}
  ADD_DEPS ON)

# This re-exports
set(arbor-builtin-mechanisms ${arbor-builtin-mechanisms} PARENT_SCOPE)

if(ARB_WITH_CUDA_CLANG OR ARB_WITH_HIP_CLANG)
    set_source_files_properties(${arbor-builtin-mechs} PROPERTIES LANGUAGE CXX)
endif()
