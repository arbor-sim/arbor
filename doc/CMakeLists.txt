include(FindPythonModule) # required for find_python_module

find_package(Sphinx)
add_target_if(SPHINX_FOUND
    check-sphinx
    "Finding Sphinx"
    "Sphinx must be installed to build documentation")

find_python_module(svgwrite)
add_target_if(HAVE_SVGWRITE
    ext-svgwrite
    "Finding svgwrite"
    "The Python module svgwrite needs to be installed to build documentation. Use pip to install svgwrite")

set(html_dir "${CMAKE_CURRENT_BINARY_DIR}/html")
set(doctree_dir "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

add_custom_target(html
    COMMAND
        PYTHONPATH="${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}"
        "${PYTHON_EXECUTABLE}"
        -m sphinx
        -b html
        -d ${doctree_dir}
        -q                          # Quiet: no output other than errors and warnings.
        ${CMAKE_CURRENT_SOURCE_DIR} # Source directory
        ${html_dir}                 # Output directory
    DEPENDS check-sphinx ext-svgwrite
    COMMENT
        "Generating Sphinx documentation")

# Remove generated documentation when make clean is run.
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${doctree_dir}" "${html_dir}")

install(DIRECTORY ${html_dir} DESTINATION ${CMAKE_INSTALL_DOCDIR} OPTIONAL)
