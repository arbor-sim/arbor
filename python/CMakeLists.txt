# Set up pybind11 as an external project.
set(pb11_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/pybind11")

include(GitSubmodule)
git_submodule(pybind11 "${pb11_src_dir}")

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
