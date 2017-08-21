# Set up rtd theme as an external project.
set(rtdtheme_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/rtd_theme")

find_package(Git)
if(NOT EXISTS "${rtdtheme_src_dir}/.git")
    set(git_failed)

    if(GIT_FOUND)
        message(STATUS "Updating the ReadTheDocs theme submodule ${rtdtheme_src_dir}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init "${rtdtheme_src_dir}"
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
        message(WARNING "Unable to update the ReadTheDocs theme submodule: ${git_failed}")
    endif()

endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/static")

# configure target for the sphinx-generated html docs
find_package(Sphinx)

set(DOCS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")
set(DOCS_DOC_TREE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")
set(DOCS_TARGET_NAME docs)
if (SPHINX_FOUND)
    add_custom_target( ${DOCS_TARGET_NAME}
        COMMAND
            ${DOCS_EXECUTABLE}
            -b html
            -d "${DOCS_DOC_TREE_DIR}"
            -q # Quiet: no output other than errors and warnings.
            "${CMAKE_CURRENT_SOURCE_DIR}" # Source
            "${DOCS_BUILD_DIR}" # Output
        COMMENT
            "Generating Sphinx documentation")
else()
    add_custom_target( ${DOCS_TARGET_NAME}
        COMMAND
            echo "Error: Sphinx must be installed to build documentation."
        COMMAND
            exit 1
        COMMENT
            "Generating Sphinx documentation")
endif()

# remove generated documentation when make clean is run
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${DOCS_BUILD_DIR}")
