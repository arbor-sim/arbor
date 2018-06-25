# Creates a target that prints an error message and returns 1 when built.
#   name    : the name of the target
#   comment : the COMMENT string for the real target, e.g. "Building the Sphinx documentation"
#   message : the error message

function(add_error_target target comment error_message)
    if(NOT TARGET "${target}")
        add_custom_target("${target}"
            COMMAND echo "  Error: ${error_message}."
            COMMAND exit 1
            COMMENT "${comment}")
    endif()
endfunction()

macro(add_target_if condition target comment error_message)
    if(${condition})
        add_custom_target("${target}"
            COMMAND true
            COMMENT "${comment}")
    else()
        add_error_target("${target}" "${comment}" "${error_message}")
    endif()
endmacro()
