include(FindPythonModule) # required for find_python_module

find_package(Sphinx)
add_target_if(SPHINX_FOUND
    check-sphinx
    "Finding Sphinx"
    "Sphinx must be installed to build documentation")

if (SPHINX_FOUND)
    find_python_module(svgwrite)
    if (NOT HAVE_SVGWRITE)
        message(WARNING "Building documentation requires the svgwrite Python module. You can obtain it from pip like so: pip3 install svgwrite --user")
    endif()
endif()

set(html_dir "${CMAKE_CURRENT_BINARY_DIR}/html")
set(doctree_dir "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

get_property(theme_path TARGET ext-sphinx_rtd_theme PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
string(REPLACE ";" "," theme_path "${theme_path}")

add_custom_target(html
    COMMAND
        ${SPHINX_EXECUTABLE}
        -b html
        -d ${doctree_dir}
        -D "html_theme_path=${theme_path}"
        -q                          # Quiet: no output other than errors and warnings.
        ${CMAKE_CURRENT_SOURCE_DIR} # Source directory
        ${html_dir}                 # Output directory
    DEPENDS check-sphinx ext-sphinx_rtd_theme
    COMMENT
        "Generating Sphinx documentation")

# Remove generated documentation when make clean is run.
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${doctree_dir}" "${html_dir}")

install(DIRECTORY ${html_dir} DESTINATION ${CMAKE_INSTALL_DOCDIR} OPTIONAL)
