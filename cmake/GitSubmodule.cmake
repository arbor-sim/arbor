# Call to ensure that the git submodule in location `path` is loaded.
# If the submodule is not loaded, an error message that describes
# how to update the submodules is printed.
# Sets the variable name_avail to `ON` if the submodule is available,
# or `OFF` otherwise.

function(check_git_submodule name path)
    set(success_var "${name}_avail")
    set(${success_var} ON PARENT_SCOPE)

    message("looking in ${path}")
    if(NOT EXISTS "${path}/.git")
        message(
            "\nThe ${name} submodule is not available.\n"
            "To check out all submodules you can use the following commands:\n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or download them when checking out:\n"
            "    git clone --recursive https://github.com/eth-cscs/arbor.git\n"
        )

        # if the repository was not available, and git failed, set AVAIL to false
        set(${success_var} OFF PARENT_SCOPE)
    endif()
endfunction()
