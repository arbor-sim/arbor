#!/usr/bin/env julia

include("LVAChannels.jl")

using JSON
using Unitful
using Unitful.DefaultSymbols
using Main.LVAChannels

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

radius = 20µm/2
area = 4*pi*radius^2
current = -0.025nA

stim = Stim(20ms, 150ms, current/area)
ts, vs, m, d, h = run_lva(300ms, param=LVAParam(vrest=-65mV), stim=stim, sample_dt=0.025ms)

trace = Dict(
    :name => "membrane voltage",
    :sim => "numeric",
    :model => "test_kinlva",
    :units => "mV",
    :data => Dict(
        :time => scale.(ts, 1ms),
        Symbol("soma.mid") => scale.(vs, 1mV)
    )
)

state = Dict(
    :name => "mechanisms state",
    :sim => "numeric",
    :model => "kinlva",
    :units => "1",
    :data => Dict(
        :time => scale.(ts, 1ms),
        Symbol("m") => m,
        Symbol("d") => d,
        Symbol("h") => h
    )
)
println(JSON.json([trace, state]))

