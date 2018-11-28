#!/usr/bin/env julia

include("NaX_Kamt_Krdmt_Channels.jl")

using JSON
using Unitful
using Unitful.DefaultSymbols
using ArgParse
using Main.spikeChannels

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gap", "-g"
            help = "turn on gap_junction"
            action = :store_true
        "--tweak", "-t"
            help = "change nax_gbar for second soma"
            action = :store_true
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

gap = parsed_args["gap"]
tweak = parsed_args["tweak"]

radius = 22.360679775µm/2
area = 4*pi*radius^2

stim = []
# Stim for soma0
push!(stim, Stim(0ms, 100ms, 0.1nA/area))
# Stim for soma1
push!(stim, Stim(10ms, 100ms, 0.1nA/area))

ts, vs = run_spike(100ms, stim=stim, sample_dt=0.025ms, gap=gap, tweak=tweak)

trace = Dict(
    :name => "membrane voltage",
    :sim => "numeric",
    :model => "soma",
    :units => "mV",
    :data => Dict(
        :time => scale.(ts, 1ms),
        Symbol("soma.mid") => scale.(vs[1], 1mV)
    )
)


trace[:data][Symbol("soma.mid"*string(2))] = scale.(vs[2], 1mV)

println(JSON.json([trace]))

