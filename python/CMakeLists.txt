# Set up pybind11 as an external project.
set(pb11_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/pybind11")

find_package(Git)
if(NOT EXISTS "${pb11_src_dir}/.git")
    set(git_failed)

    if(GIT_FOUND)
        message(STATUS "Updating the Pybind11 submodule ${pb11_src_dir}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init "${pb11_src_dir}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            ERROR_VARIABLE git_error
            RESULT_VARIABLE git_result)
        if(NOT git_result EQUAL 0)
            set(git_failed "${git_error}")
        endif()
    else()
        set(git_failed "git not found")
    endif()

    if(git_failed)
        message(WARNING "Unable to update the Pybind11 submodule: ${git_failed}")
    endif()

endif()

# Set up pybind11, which is used to generate python bindings.
# We can just add the pybind11 directory, instead of find_package().
set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(pybind11)

pybind11_add_module(pyarb pyarb.cpp sampling.cpp)

target_link_libraries(pyarb LINK_PUBLIC ${ARB_LIBRARIES})
target_link_libraries(pyarb LINK_PUBLIC ${EXTERNAL_LIBRARIES})

if(ARB_WITH_MPI)
    target_link_libraries(pyarb LINK_PUBLIC ${MPI_C_LIBRARIES})
    set_property(TARGET pyarb APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()
