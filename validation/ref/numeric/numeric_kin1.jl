#!/usr/bin/env julia

include("HHChannels.jl")

using JSON
using SIUnits.ShortUnits

radius = 20µm/2
area = 4*pi*radius^2
sample_dt = 0.025ms
t_end = 100ms

a0 = 0.01mA/cm^2
c  = 0.01mA/cm^2

tau = 10ms

ts = collect(0s: sample_dt: t_end)
is = area*(1/3*c + (a0-1/3*c)*exp(-ts/tau))

trace = Dict(
    :name => "membrane current",
    :sim => "numeric",
    :model => "test_kin1",
    :units => "nA",
    :data => Dict(
        :time => map(t->t/ms, ts),
        Symbol("soma.mid") => map(i->i/nA, is)
    )
)

println(JSON.json([trace]))

