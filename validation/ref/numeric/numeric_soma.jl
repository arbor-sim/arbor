#!/usr/bin/env julia

include("HHChannels.jl")

using JSON
using Unitful
using Unitful.DefaultSymbols
using Main.HHChannels

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

radius = 18.8µm/2
area = 4*pi*radius^2

stim = Stim(10ms, 100ms, 0.1nA/area)
ts, vs = run_hh(100ms, stim=stim, sample_dt=0.025ms)

trace = Dict(
    :name => "membrane voltage",
    :sim => "numeric",
    :model => "soma",
    :units => "mV",
    :data => Dict(
        :time => scale.(ts, 1ms),
        Symbol("soma.mid") => scale.(vs, 1mV)
    )
)

println(JSON.json([trace]))

