# note: function add_validation_data defined in validation/CMakeLists.txt

set(models
     ball_and_stick
     ball_and_3stick
     ball_and_squiggle
     ball_and_taper
     simple_exp_synapse
     simple_exp2_synapse
     soma)

foreach(model ${models})
    set(script "${model}.py")
    add_validation_data(
        OUTPUT "neuron_${model}.json"
        DEPENDS "${script}" "nrn_validation.py"
        COMMAND ${Nrniv_EXECUTABLE} -nobanner -python "${script}")
endforeach()

