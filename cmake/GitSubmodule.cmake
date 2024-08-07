# Call to ensure that the git submodule in location `path` is loaded.
# If the submodule is not loaded, an error message that describes
# how to update the submodules is printed.
# Sets the variable name_avail to `ON` if the submodule is available,
# or `OFF` otherwise.

function(check_git_submodule name path)
    set(success_var "${name}_avail")
    set(${success_var} ON PARENT_SCOPE)

    get_filename_component(dotgit "${path}/.git" ABSOLUTE)
    if(NOT EXISTS ${dotgit})
        message(
            "\nThe git submodule for ${name} is not available.\n"
            "To check out all submodules use the following commands:\n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or download submodules recursively when checking out:\n"
            "    git clone --recursive https://github.com/arbor-sim/arbor.git\n"
        )
        get_filename_component(dotgit "${path}/" ABSOLUTE)
        file(GLOB RESULT "${path}/*")
        list(LENGTH RESULT RES_LEN)
        if(NOT RES_LEN EQUAL 0)
            message("However, the directory contains some ($RES_LEN) files. Possibly .git was omitted?")
        endif()
        # if the repository was not available, and git failed, set AVAIL to false
        set(${success_var} OFF PARENT_SCOPE)
    endif()
endfunction()
