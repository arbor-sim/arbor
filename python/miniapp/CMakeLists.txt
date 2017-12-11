set(PYMINIAPP_SOURCES
  miniapp-base.cpp
  io.cpp
  miniapp_recipes.cpp
  morphology_pool.cpp
  trace.cpp
)

add_cython_target( pyminiapp CXX )
add_library( pyminiapp MODULE ${pyminiapp} ${PYMINIAPP_SOURCES} )
#python_extension_module(pyminiapp)

install(TARGETS pyminiapp LIBRARY DESTINATION pyminiapp)

target_link_libraries(pyminiapp ${ARB_LIBRARIES})
target_link_libraries(pyminiapp ${EXTERNAL_LIBRARIES})

include_directories("${PROJECT_SOURCE_DIR}/miniapp")
target_link_libraries(pyminiapp ${PYTHON_LIBRARIES})

if(ARB_WITH_MPI)
    target_link_libraries(pyminiapp ${MPI_C_LIBRARIES})
    set_property(TARGET pyminiapp APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

set_target_properties(
    pyminiapp
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/miniapps/pyminiapp"
)
