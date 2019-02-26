# note: function add_validation_data defined in validation/CMakeLists.txt

add_validation_data(
    OUTPUT numeric_kin1.json
    DEPENDS numeric_kin1.jl
    COMMAND ${Julia_EXECUTABLE} numeric_kin1.jl)

add_validation_data(
    OUTPUT numeric_kinlva.json
    DEPENDS numeric_kinlva.jl LVAChannels.jl
    COMMAND ${Julia_EXECUTABLE} numeric_kinlva.jl)

add_validation_data(
    OUTPUT numeric_soma.json
    DEPENDS numeric_soma.jl HHChannels.jl
    COMMAND ${Julia_EXECUTABLE} numeric_soma.jl)

add_validation_data(
    OUTPUT numeric_rallpack1.json
    DEPENDS numeric_rallpack1.jl PassiveCable.jl
    COMMAND ${Julia_EXECUTABLE} numeric_rallpack1.jl)
