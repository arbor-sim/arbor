# Set up pybind11 as an external project.
set(pb11_src_dir "${PROJECT_SOURCE_DIR}/python/pybind11")
check_git_submodule(pybind11 "${pb11_src_dir}")

if(NOT pybind11_avail)
    message(FATAL_ERROR "The git submodule for pybind11 is not available: unable to support python"
endif()

# Set up pybind11, which is used to generate python bindings.
# Pybind11 has good cmake support, so just add the pybind11 directory,
# instead of using find_package.
set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(pybind11)

pybind11_add_module(pyarb pyarb.cpp)

target_link_libraries(pyarb LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(pyarb LINK_PUBLIC ${EXTERNAL_LIBRARIES})

if(ARB_WITH_MPI)
    target_link_libraries(pyarb LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET pyarb APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()
