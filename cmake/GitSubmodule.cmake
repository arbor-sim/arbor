function(git_submodule name path)
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
        endif()

    endif()
endfunction()
