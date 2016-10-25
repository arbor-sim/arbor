# The validation scripts to run (without .py extension)

set(validations
     ball_and_stick
     ball_and_3stick
     ball_and_taper
     simple_exp_synapse
     simple_exp2_synapse
     soma)

# Only try and make validation sets if we can find nrniv
if(BUILD_VALIDATION_DATA)
    set(common "${CMAKE_CURRENT_SOURCE_DIR}/nrn_validation.py")
    foreach(v ${validations})
	set(out "${CMAKE_SOURCE_DIR}/data/validation/neuron_${v}.json")
	set(src "${CMAKE_CURRENT_SOURCE_DIR}/${v}.py")
	add_custom_command(
	    OUTPUT "${out}"
	    DEPENDS "${src}" "${common}"
	    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
	    COMMAND ${NRNIV_BIN} -nobanner -python ${src} > ${out})
        list(APPEND all_neuron_validation "${out}")
    endforeach()
    add_custom_target(validation_data DEPENDS ${all_neuron_validation})
endif()

