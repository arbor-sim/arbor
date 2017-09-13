include(CMakeParseArguments)

# Uses CMake variables modcc and use_external_modcc as set in top level CMakeLists.txt

function(build_modules)
    cmake_parse_arguments(build_modules "" "TARGET;SOURCE_DIR;DEST_DIR;MECH_SUFFIX" "MODCC_FLAGS" ${ARGN})

    foreach(mech ${build_modules_UNPARSED_ARGUMENTS})
        set(mod "${build_modules_SOURCE_DIR}/${mech}.mod")
        set(out "${build_modules_DEST_DIR}/${mech}")

        set(depends "${mod}")
        if(NOT use_external_modcc)
            list(APPEND depends modcc)
        endif()

        set(flags ${build_modules_MODCC_FLAGS} -o "${out}")
        if(build_modules_MECH_SUFFIX)
            list(APPEND flags -m "${mech}${build_modules_MECH_SUFFIX}")
        endif()

        add_custom_command(
            OUTPUT ${out}.hpp ${out}_impl.hpp ${out}_impl.cu
            DEPENDS ${depends}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMAND ${modcc} ${flags} ${mod}
        )
        set_source_files_properties("${out}.hpp"      PROPERTIES GENERATED TRUE)
        set_source_files_properties("${out}_impl.hpp" PROPERTIES GENERATED TRUE)
        set_source_files_properties("${out}_impl.cu"  PROPERTIES GENERATED TRUE)
        list(APPEND all_mod_hpps ${out}.hpp ${out}_impl.hpp ${out}_impl.cu)
    endforeach()

    # Fake target to always trigger .mod -> .hpp/.cu dependencies because CMake
    if (build_modules_TARGET)
        set(depends ${all_mod_hpps})
        if(NOT use_external_modcc)
            list(APPEND depends modcc)
        endif()
        add_custom_target(${build_modules_TARGET} DEPENDS ${depends})
    endif()
endfunction()
