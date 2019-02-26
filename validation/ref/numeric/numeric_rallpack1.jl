#!/usr/bin/env julia

include("PassiveCable.jl")

using JSON
using Unitful
using Unitful.DefaultSymbols
using Main.PassiveCable

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

# This should run the same effective model
# as rallpack1, but with differing
# electrical parameters (see below).

function run_cable(x_prop, ts)
    # Physical properties:

    # f is a fudge factor. rM needs to be the same
    # the same as in arbor, where we cannot yet set
    # the membrane conductance parameter. Scaling
    # other parameters proportionally, however,
    # gives the same dynamics.

    f = 0.1/4

    diam =   1.0µm        # cable diameter
    L    =   1.0mm        # cable length
    I    =   0.1nA     /f # current injection
    rL   =   1.0Ω*m    *f # bulk resistivity
    erev = -65.0mV        # (passive) reversal potential
    rM   =   4Ω*m^2    *f # membrane resistivity
    cM   =   0.01F/m^2 /f # membrane specific capacitance

    # convert to linear resistivity, length and time constants
    area = pi*diam^2/4
    r    = rL/area

    lambda = sqrt(diam/4 * rM/rL)
    tau = cM*rM

    # compute solutions
    tol = 1e-8mV
    return [cable(L*x_prop, t, L, lambda, tau, r, erev, -I, tol=tol) for t in ts]
end

function run_rallpack1(x_prop, ts)
    return [rallpack1(0.001*x_prop, scale(t, 1s))*V for t in ts]
end

# Generate traces at x=0, x=0.3L, x=L

ts = collect(0s: 0.025ms: 250ms)

trace = Dict(
    :name => "membrane voltage",
    :sim => "numeric",
    :model => "rallpack1",
    :units => "mV",
    :data => Dict(
        :time => scale.(ts, 1ms),
        Symbol("cable.x0.0") => scale.(run_cable(0, ts), 1mV),
        Symbol("cable.x0.3") => scale.(run_cable(0.3, ts), 1mV),
        Symbol("cable.x1.0") => scale.(run_cable(1.0, ts), 1mV)
    )
)

 println(JSON.json([trace]))

