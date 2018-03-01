include("${CMAKE_SOURCE_DIR}/cmake/GitSubmodule.cmake")

# Set up rtd theme as an external project.
set(rtdtheme_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/rtd_theme")

find_package(Git)
git_submodule(ReadTheDocs "${rtdtheme_src_dir}")

# a static path is required to avoid warning messages from sphinx-build
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/static")

# configure target for the sphinx-generated html docs
find_package(Sphinx)

set(DOCS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")
set(DOCS_DOC_TREE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")
set(DOCS_TARGET_NAME docs)
if (SPHINX_FOUND)
    add_custom_target( ${DOCS_TARGET_NAME}
        COMMAND
            ${SPHINX_EXECUTABLE}
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
            exit 1  # return error code to let CMake know that the build proccess should fail
        COMMENT
            "Generating Sphinx documentation")
endif()

# remove generated documentation when make clean is run
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${DOCS_BUILD_DIR}")

unset(git_failed)
