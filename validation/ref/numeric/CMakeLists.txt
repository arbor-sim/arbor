# note: function add_validation_data defined in validation/CMakeLists.txt

add_validation_data(
    OUTPUT numeric_soma.json
    DEPENDS numeric_soma.jl HHChannels.jl
    COMMAND julia numeric_soma.jl)

add_validation_data(
    OUTPUT numeric_rallpack1.json
    DEPENDS numeric_rallpack1.jl PassiveCable.jl
    COMMAND julia numeric_rallpack1.jl)
