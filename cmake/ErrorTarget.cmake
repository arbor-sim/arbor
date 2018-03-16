# Creates a target that prints an error message and returns 1 when built.
#   name    : the name of the target
#   comment : the COMMENT string for the real target, e.g. "Building the Sphinx documentation"
#   message : the error message

function(add_error_target name comment message)
    add_custom_target(${name}
        COMMAND echo "  Error: ${message}."
        COMMAND exit 1
        COMMENT "${comment}")
endfunction()
