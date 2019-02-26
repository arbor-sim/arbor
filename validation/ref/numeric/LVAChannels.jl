module LVAChannels

export Stim, run_lva, LVAParam

using Sundials
using Unitful
using Unitful.DefaultSymbols

struct LVAParam
    c_m       # membrane spacific capacitance
    gbar      # Ca channel cross-membrane conductivity
    eca       # Ca channel reversal potential
    gl        # leak conductivity
    el        # leak reversal potential
    q10_1     # rate scaling for 'm' activation gate
    q10_2     # rate scaling for 'dsh' inactivation gate
    vrest     # (as hoc) resting potential

    # constructor with default values, with q10 values
    # corresponding to the body temperature current-clamp experiments
    LVAParam(;
        c_m    = 0.01F*m^-2,
        gbar   = 0.2mS*cm^-2,
        eca    = 120mV,
        gl     = .1mS*cm^-2,
        el     = -65mV,
        q10_1  = 5,
        q10_2  = 3,
#        vrest  = -61.47mV
        vrest  = -63mV
    ) = new(c_m, gbar, eca, gl, el, q10_1, q10_2, vrest)

end

struct Stim
    t0        # start time of stimulus
    t1        # stop time of stimulus
    i_e       # stimulus current density

    Stim() = new(0s, 0s, 0A/m^2)
    Stim(t0, t1, i_e) = new(t0, t1, i_e)
end

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

# 'm' activation gate
function m_lims(v, q10)
    quotient = 1+exp(-(v+63mV)/7.8mV)
    mtau = 1ms*(1.7+exp(-(v+28.8mV)/13.5mV))/(q10*quotient)
    minf = 1/quotient
    return mtau, minf
end

# 'dsh' inactivation gate:
# d <-> s (alpha2, beta2)
# s <-> h (alpha1, beta1)
# subject to s=1-h-d
function dsh_lims(v, q10)
    k = sqrt(0.25+exp((v+83.5mV)/6.3mV))-0.5
    alpha1 = q10*exp(-(v+160.3mV)/17.8mV)/1ms
    beta1 = alpha1*k

    tau2 = 240ms/(1+exp((v+37.4mV)/30mV))/q10
    alpha2 = 1/tau2/(1+k)
    beta2 = alpha2*k

    hinf = 1/(1+k+k^2)
    dinf = k^2*hinf

    return alpha1, beta1, alpha2, beta2, dinf, hinf
end

# Choose initial conditions for the system such that the gating variables
# are at steady state for the user-specified voltage v.
function initial_conditions(v, q10_1, q10_2)
    mtau, minf = m_lims(v, q10_1)
    alpha1, beta1, alpha2, beta2, dinf, hinf = dsh_lims(v, q10_2)

    return (v, minf, dinf, hinf)
end

# Given time t and state (v, m, d, h),
# return (vdot, mdot, ddot, hdot).
function f(t, state; p=LVAParam(), istim=0A/m^2)
    v, m, d, h = state

    ica = p.gbar*m^3*h*(v - p.eca)
    il = p.gl*(v - p.el)
    itot = ica + il - istim

    # Calculate the voltage dependent rates for the gating variables.
    mtau, minf = m_lims(v, p.q10_1)
    alpha1, beta1, alpha2, beta2, dinf, hinf = dsh_lims(v, p.q10_2)

    mdot = (minf-m)/mtau
    hdot = alpha1*(1-h-d)-beta1*h
    ddot = beta2*(1-h-d)-alpha2*d

    return (-itot/p.c_m, mdot, ddot, hdot)
end

function make_range(t_start, dt, t_end)
    r = collect(t_start: dt: t_end)
    if length(r)>0 && r[length(r)]<t_end
        push!(r, t_end)
    end
    return r
end

function run_lva(t_end; stim=Stim(), param=LVAParam(), sample_dt=0.01ms)
    v_scale = 1V
    t_scale = 1s

    v0, m0, h0, d0 = initial_conditions(param.vrest, param.q10_1, param.q10_2)
    y0 = [ scale(v0, v_scale), m0, h0, d0 ]


    fbis(t, y, ydot, istim) = begin
        vdot, mdot, hdot, ddot =
            f(t*t_scale, (y[1]*v_scale, y[2], y[3], y[4]), istim=istim, p=param)

        ydot[1], ydot[2], ydot[3], ydot[4] =
            vdot*t_scale/v_scale, mdot*t_scale, hdot*t_scale, ddot*t_scale

        return Sundials.CV_SUCCESS
    end

    fbis_nostim(t, y, ydot) = fbis(t, y, ydot, 0A/m^2)
    fbis_stim(t, y, ydot) = fbis(t, y, ydot, stim.i_e)

    # Ideally would run with vector absolute tolerance to account for v_scale,
    # but this would prevent us using the nice cvode wrapper.

    res = []
    samples = []

    t1 = clamp(stim.t0, 0s, t_end)
    if t1>0s
        ts = make_range(0s, sample_dt, t1)
        r = Sundials.cvode(fbis_nostim, y0, scale.(ts, t_scale), abstol=1e-6, reltol=5e-10)
        y0 = vec(r[size(r)[1], :])
        push!(res, r)
        push!(samples, ts)
    end
    t2 = clamp(stim.t1, t1, t_end)
    if t2>t1
        ts = make_range(t1, sample_dt, t2)
        r = Sundials.cvode(fbis_stim, y0, scale.(ts, t_scale), abstol=1e-6, reltol=5e-10)
        y0 = vec(r[size(r)[1], :])
        push!(res, r)
        push!(samples, ts)
    end
    if t_end>t2
        ts = make_range(t2, sample_dt, t_end)
        r = Sundials.cvode(fbis_nostim, y0, scale.(ts, t_scale), abstol=1e-6, reltol=5e-10)
        y0 = vec(r[size(r)[1], :])
        push!(res, r)
        push!(samples, ts)
    end

    res = vcat(res...)
    samples = vcat(samples...)

    return samples, res[:, 1]*v_scale, res[:, 2], res[:, 3], res[:, 4]
end

end # module LVAChannels
