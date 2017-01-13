# note: function add_validation_data defined in validation/CMakeLists.txt

if(NMC_BUILD_JULIA_VALIDATION_DATA)
    add_validation_data(
        OUTPUT numeric_soma.json
        DEPENDS numeric_soma.jl HHChannels.jl
        COMMAND ${JULIA_BIN} numeric_soma.jl)

    add_validation_data(
        OUTPUT numeric_rallpack1.json
        DEPENDS numeric_rallpack1.jl PassiveCable.jl
        COMMAND ${JULIA_BIN} numeric_rallpack1.jl)
endif()
