include(CMakeParseArguments)

# If a MODCC executable is explicitly provided, don't make the in-tree modcc a dependency.

function(build_modules)
    cmake_parse_arguments(build_modules "" "MODCC;TARGET;SOURCE_DIR;DEST_DIR;MECH_SUFFIX" "MODCC_FLAGS;GENERATES" ${ARGN})

    if("${build_modules_SOURCE_DIR}" STREQUAL "")
        set(build_modules_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    if("${build_modules_DEST_DIR}" STREQUAL "")
        set(build_modules_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    file(MAKE_DIRECTORY "${build_modules_DEST_DIR}")

    set(all_generated)
    foreach(mech ${build_modules_UNPARSED_ARGUMENTS})
        set(mod "${build_modules_SOURCE_DIR}/${mech}.mod")
        set(out "${build_modules_DEST_DIR}/${mech}")
        set(generated)
        foreach (suffix ${build_modules_GENERATES})
            list(APPEND generated ${out}${suffix})
        endforeach()

        set(depends "${mod}")
        if(build_modules_MODCC)
            set(modcc_bin ${build_modules_MODCC})
        else()
            list(APPEND depends modcc)
            set(modcc_bin $<TARGET_FILE:modcc>)
        endif()

        set(flags ${build_modules_MODCC_FLAGS} -o "${out}")
        if(build_modules_MECH_SUFFIX)
            list(APPEND flags -m "${mech}${build_modules_MECH_SUFFIX}")
        endif()

        add_custom_command(
            OUTPUT ${generated}
            DEPENDS ${depends}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMAND ${modcc_bin} ${flags} ${mod}
            COMMENT "modcc generating: ${generated}"
        )
        set_source_files_properties(${generated} PROPERTIES GENERATED TRUE)
        list(APPEND all_generated ${generated})
    endforeach()

    # Fake target to always trigger .mod -> .hpp/.cu dependencies because CMake
    if (build_modules_TARGET)
        set(depends ${all_generated})
        if(NOT build_modules_MODCC)
            list(APPEND depends modcc)
        endif()
        add_custom_target(${build_modules_TARGET} DEPENDS ${depends})
    endif()
endfunction()
