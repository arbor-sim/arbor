set(MINIAPP_SOURCES
    miniapp.cpp
    io.cpp
    miniapp_recipes.cpp
    morphology_pool.cpp
    trace.cpp
    )

# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
  ${PYARBOR_MINIAPP_SOURCE_DIR}/src/pyarbor_miniapp.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( pyarbor_miniapp ${MINIAPP_SOURCES} pyarbor_miniapp.pyx )

#target_link_libraries(miniapp.exe LINK_PUBLIC ${ARB_LIBRARIES})
#target_link_libraries(miniapp.exe LINK_PUBLIC ${EXTERNAL_LIBRARIES})

#if(ARB_WITH_MPI)
#    target_link_libraries(miniapp.exe LINK_PUBLIC ${MPI_C_LIBRARIES})
#    set_property(TARGET miniapp.exe APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
#endif()

#set_target_properties(
#    miniapp.exe
#    PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/miniapps/miniapp"
#)
