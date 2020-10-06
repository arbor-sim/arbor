import os, sys
# Path to Python Binding (_arbor)
try:
    autodoc_output_file = this_path=os.path.join(os.path.split(os.path.abspath(__file__))[0],'..','py_reference.rst')

    # Add the local build directory to where Python searches for Arbor.

    sys.path.append(os.path.join(os.environ['OLDPWD'],"python"))
    # Only generate a fresh autodoc_output_file if there is an importable Arbor package
    import arbor

    print("--- generating autodoc cache ---")

    # Generate title such that the page shows up in Sphinx.
    with open(autodoc_output_file, "w") as file_object:
        file_object.write('Python API reference\n')
        file_object.write('====================\n')

    # Override add_line and intercept intermediate rst output. Replace arbor._arbor while we're at it
    import sphinx.ext.autodoc
    def add_line(self, line, source, *lineno):
        """Append one line of generated reST to the output."""
        line = line.replace('arbor._arbor','arbor')
        with open(autodoc_output_file, "a") as file_object:
            file_object.write(self.indent + line + '\n')
        self.directive.result.append(self.indent + line, source, *lineno)

    sphinx.ext.autodoc.Documenter.add_line = add_line

except ImportError:
    # If not package here, hope autodoc_output_file is already checked in.
    # Setup mock imports to stop autodoc from complaining about a missing package.
    autodoc_mock_imports = ['arbor._arbor']