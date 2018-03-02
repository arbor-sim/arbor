
# Call to ensure that the git submodule in location `path` is loaded.
# If the submodule is not loaded, the function will attempt to check it
# out using git.
# Sets the following variable `name_AVAIL` to `ON` if the submodule is
# available on completion, or `OFF` otherwise.

function(git_submodule name path)
    set(success_var "${name}_AVAIL")
    set(${success_var} ON PARENT_SCOPE)

    if(NOT EXISTS "${path}/.git")
        set(git_failed)

        if(GIT_FOUND)
            message(STATUS "Updating the ${name} theme submodule ${rtdtheme_src_dir}")
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" submodule update --init "${path}"
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
            message(WARNING "Unable to update the ${name} theme submodule: ${git_failed}")
            # if the repository was not available, and git failed, set AVAIL to false
            set(${success_var} OFF PARENT_SCOPE)
        endif()
    endif()
endfunction()
