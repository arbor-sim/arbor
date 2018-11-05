#!/usr/bin/env julia

include("HHChannels_multipleSomas.jl")

using JSON
using Unitful
using Unitful.DefaultSymbols
using ArgParse
using DataStructures
using Main.HHChannels

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n_cells", "-n"
            help = "number of cells"
            arg_type = Int
            default = 3
        "--diameter", "-d"
            help = "diameter of cells in µm"
            arg_type = Float64
            default = 18.8
        "--ggap", "-g"
            help = "conductivity of gap junctions in µS"
            arg_type = Float64
            default = 5e-5
        "--stim_t0"
            help = "start time of stimulus in ms"
            arg_type = Float64
            default = 20.0
        "--stim_t1"
            help = "end time of stimulus in ms"
            arg_type = Float64
            default = 22.0
        "--stim", "-s"
            help = "strength of stimulus in nA"
            arg_type = Float64
            default = 0.1
        "--t_end", "-t"
            help = "simulation time in ms"
            arg_type = Float64
            default = 100.0
        "--dt"
            help = "time step in ms"
            arg_type = Float64
            default = 0.025
        "--output", "-o"
            help = "JSON filename"
            default = "output.json"
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

n_cells = parsed_args["n_cells"]
radius = parsed_args["diameter"]/2*µm
area = 4*pi*radius^2

ggap = parsed_args["ggap"]*1e-6/scale(area, cm^2)*S*cm^-2

ts0 = parsed_args["stim_t0"]*ms
ts1 = parsed_args["stim_t1"]*ms
St  = parsed_args["stim"]/area*nA

stim = Stim(ts0, ts1, St)

t_end = parsed_args["t_end"]*ms
dt = parsed_args["dt"]*ms

ts, vs = run_hh(t_end, ggap, n_cells, stim=stim, sample_dt=dt)

trace = Dict(
    :name => "membrane voltage",
    :sim => "numeric",
    :model => "soma",
    :units => "mV",
    :data => OrderedDict{}(
        :time => scale.(ts, 1ms),
        Symbol("soma1.mid") => scale.(vs[1], 1mV)
    )
)

for i=2:n_cells
    trace[:data][Symbol("soma.mid"*string(i))] = scale.(vs[i], 1mV)
end

fname = "output_$(parsed_args["ggap"]).txt"
io = open(fname, "w");
println(io, JSON.json([trace]))
