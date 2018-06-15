find_package(Sphinx)
if(NOT SPHINX_FOUND)
    add_error_target(docs
        "Generating Sphinx documentation"
        "Sphinx must be installed to build documentation")
endif()

check_git_submodule(rtdtheme "${PROJECT_SOURCE_DIR}/doc/rtd_theme")
if (NOT rtdtheme_avail)
    add_error_target(docs
        "Generating Sphinx documentation"
        "The git submodule for read the docs is not available")
endif()

if(NOT TARGET docs)
    # A static path is required to avoid warning messages from sphinx-build.
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/static")

    set(html_dir "${CMAKE_CURRENT_BINARY_DIR}/html")
    set(doctree_dir "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

    add_custom_target(docs
        COMMAND
            ${SPHINX_EXECUTABLE}
            -b html
            -d "${doctree_dir}"
            -q # Quiet: no output other than errors and warnings.
            "${CMAKE_CURRENT_SOURCE_DIR}" # Source
            "${html_dir}" # Output
        COMMENT
            "Generating Sphinx documentation")

    # Remove generated documentation when make clean is run.
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${doctree_dir}" "${html_dir}")

    install(DIRECTORY ${html_dir} DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()

