include(CMakeParseArguments)

function("make_catalogue")
  cmake_parse_arguments(MK_CAT "" "NAME;SOURCES;VERBOSE" "CXX_FLAGS_TARGET;MOD;CXX" ${ARGN})
  set(MK_CAT_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/${MK_CAT_NAME}")
  file(MAKE_DIRECTORY "${MK_CAT_OUT_DIR}")

  if(MK_CAT_VERBOSE)
    message("Catalogue name:       ${MK_CAT_NAME}")
    message("Catalogue mechanisms: ${MK_CAT_MOD}")
    message("Extra cxx files:      ${MK_CAT_CXX}")
    message("Catalogue sources:    ${MK_CAT_SOURCES}")
    message("Catalogue output:     ${MK_CAT_OUT_DIR}")
    message("Arbor cxx flags:      ${MK_CAT_CXX_FLAGS_TARGET}")
    message("Arbor cxx compiler:   ${ARB_CXX}")
    message("Current cxx compiler: ${CMAKE_CXX_COMPILER}")
  endif()

  set(mk_cat_modcc_flags -t cpu ${ARB_MODCC_FLAGS} -N arb -c ${MK_CAT_NAME} -o ${MK_CAT_OUT_DIR})
  if(ARB_WITH_GPU)
    set(mk_cat_modcc_flags, -t gpu ${mk_cat_modcc_flags})
  endif()

  list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${MK_CAT_NAME}_catalogue.cpp)

  foreach(mech ${MK_CAT_MOD})
    list(APPEND catalogue_${MK_CAT_NAME}_mods ${MK_CAT_SOURCES}/${mech}.mod)
    list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${mech}_cpu.cpp)
    if(ARB_WITH_GPU)
      list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${mech}_gpu.cpp ${MK_CAT_OUT_DIR}/${mech}_gpu.cu)
    endif()
  endforeach()

  foreach(mech ${MK_CAT_CXX})
    list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${mech}_cpu.cpp)
    if(ARB_WITH_GPU)
      list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${mech}_gpu.cpp ${MK_CAT_OUT_DIR}/${mech}_gpu.cu)
    endif()
  endforeach()

  add_custom_command(OUTPUT            ${catalogue_${MK_CAT_NAME}_source}
                     DEPENDS           ${modcc} ${catalogue_${MK_CAT_NAME}_mods}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     COMMAND ${modcc} ${mk_cat_modcc_flags} ${catalogue_${MK_CAT_NAME}_mods}
                     COMMENT "modcc generating: ${catalogue_${MK_CAT_NAME}_source}")
  add_custom_target(catalogue-${MK_CAT_NAME}-target DEPENDS ${catalogue_${MK_CAT_NAME}_source})

  set(catalogue-${MK_CAT_NAME}-mechs ${catalogue_${MK_CAT_NAME}_source} PARENT_SCOPE)

endfunction()

include(CMakeParseArguments)

function("make_catalogue_standalone")
  cmake_parse_arguments(MK_CAT "" "NAME;SOURCES;PREFIX;VERBOSE" "CXX_FLAGS_TARGET;MOD;CXX" ${ARGN})
  set(MK_CAT_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/${MK_CAT_NAME}")
  file(MAKE_DIRECTORY "${MK_CAT_OUT_DIR}")

  if(MK_CAT_VERBOSE)
    message("Catalogue name:       ${MK_CAT_NAME}")
    message("Catalogue mechanisms: ${MK_CAT_MOD}")
    message("Extra cxx files:      ${MK_CAT_CXX}")
    message("Catalogue sources:    ${MK_CAT_SOURCES}")
    message("Catalogue output:     ${MK_CAT_OUT_DIR}")
    message("Arbor cxx flags:      ${MK_CAT_CXX_FLAGS_TARGET}")
    message("Arbor cxx compiler:   ${ARB_CXX}")
    message("Current cxx compiler: ${CMAKE_CXX_COMPILER}")
  endif()

  set(mk_cat_modcc_flags -t cpu ${ARB_MODCC_FLAGS} -N arb -c ${MK_CAT_NAME} -o ${MK_CAT_OUT_DIR})
  if(ARB_WITH_GPU)
    set(mk_cat_modcc_flags, -t gpu ${mk_cat_modcc_flags})
  endif()

  set(catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${MK_CAT_NAME}_catalogue.cpp)

  foreach(mech ${MK_CAT_MOD})
    list(APPEND catalogue_${MK_CAT_NAME}_mods ${MK_CAT_SOURCES}/${mech}.mod)
    list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${mech}_cpu.cpp)
    if(ARB_WITH_GPU)
      list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${mech}_gpu.cpp ${MK_CAT_OUT_DIR}/${mech}_gpu.cu)
    endif()
  endforeach()

  add_custom_command(OUTPUT            ${catalogue_${MK_CAT_NAME}_source}
                     DEPENDS           ${catalogue_${MK_CAT_NAME}_mods}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     COMMAND ${modcc} ${mk_cat_modcc_flags} ${catalogue_${MK_CAT_NAME}_mods}
                     COMMENT "modcc generating: ${catalogue_${MK_CAT_NAME}_source}")

  foreach(mech ${MK_CAT_CXX})
    list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${mech}_cpu.cpp)
    if(ARB_WITH_GPU)
      list(APPEND catalogue_${MK_CAT_NAME}_source ${MK_CAT_OUT_DIR}/${mech}_gpu.cpp ${MK_CAT_OUT_DIR}/${mech}_gpu.cu)
    endif()
  endforeach()

  add_library(${MK_CAT_NAME}-catalogue SHARED ${catalogue_${MK_CAT_NAME}_source})
  target_compile_definitions(${MK_CAT_NAME}-catalogue PUBLIC STANDALONE=1)

  if(ARB_WITH_GPU)
    target_compile_definitions(${MK_CAT_NAME}-catalogue PUBLIC ARB_GPU_ENABLED)
  endif()

  target_compile_options(${MK_CAT_NAME}-catalogue PUBLIC ${MK_CAT_CXX_FLAGS_TARGET})
  set_target_properties(${MK_CAT_NAME}-catalogue
    PROPERTIES
    SUFFIX ".so"
    PREFIX ""
    CXX_STANDARD 17)

  if(TARGET arbor)
    target_link_libraries(${MK_CAT_NAME}-catalogue PRIVATE arbor)
  else()
    target_link_libraries(${MK_CAT_NAME}-catalogue PRIVATE arbor::arbor)
  endif()
endfunction()
