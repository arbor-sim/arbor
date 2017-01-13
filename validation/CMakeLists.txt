# Validation data generation

add_custom_target(validation_data)

# Helper function because ffs CMake.

function(make_unique_target_name name path)
    # Try and make a broadly human readable target name if possible.
    string(REGEX REPLACE ".*/" "" leaf "${path}")
    string(REGEX REPLACE "[^a-zA-Z0-9_.+-]" "_" canon "${leaf}")

    # Check against reserved names, of which of course there is no documented list
    # except in the sodding CMake source code. Seriously. Look at the CMP0037 policy
    # text for a laugh.
    if(canon MATCHES "^(all|ALL_BUILD|help|install|INSTALL|preinstall|clean|edit_cache|rebuild_cache|test|RUN_TESTS|package|PACKAGE|package_source|ZERO_CHECK)$")
        set(canon "${canon}_")
    endif()
    while((TARGET "${canon}"))
        set(canon "${canon}_")
    endwhile()
    set("${name}" "${canon}" PARENT_SCOPE)
endfunction()

# Helper function to add a data generation script that writes to standard output.
# e.g.:
#     add_validation_data(OUTPUT foo_model.json DEPENDS foo_model.py common.py COMMAND python foo_model.py)

include(CMakeParseArguments)
function(add_validation_data)
    cmake_parse_arguments(ADD_VALIDATION_DATA "" "OUTPUT" "DEPENDS;COMMAND" ${ARGN})
    set(out "${NMC_VALIDATION_DATA_DIR}/${ADD_VALIDATION_DATA_OUTPUT}")
    string(REGEX REPLACE "([^;]+)" "${CMAKE_CURRENT_SOURCE_DIR}/\\1" deps "${ADD_VALIDATION_DATA_DEPENDS}")
    add_custom_command(
        OUTPUT "${out}"
        DEPENDS ${deps}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMAND ${ADD_VALIDATION_DATA_COMMAND} > "${out}")

    # Cmake, why can't we just write add_dependencies(validation_data "${out}")?!
    make_unique_target_name(ffs_cmake "${out}")
    add_custom_target("${ffs_cmake}" DEPENDS "${out}")
    add_dependencies(validation_data "${ffs_cmake}")
endfunction()


if(NMC_BUILD_NRN_VALIDATION_DATA)
    add_subdirectory(ref/neuron)
endif()

add_subdirectory(ref/numeric)

