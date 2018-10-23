module HHChannels

export Stim, run_hh

using Sundials
using Unitful
using Unitful.DefaultSymbols

struct HHParam
    c_m       # membrane spacific capacitance
    gnabar    # Na channel cross-membrane conductivity
    gkbar     # K channel cross-membrane conductivity
    gl        # Leak conductivity
    ena       # Na channel reversal potential
    ek        # K channel reversal potential
    el        # Leak reversal potential
    q10       # temperature dependent rate coefficient
              # (= 3^((T-T₀)/10K) with T₀ = 6.3 °C)

    # constructor with default values, corresponding
    # to a resting potential of -65 mV and temperature 6.3 °C
    HHParam(;
        # default values from HH paper

        # For reversal potentials we use those computed using
        # the Nernst equation with the following values:
        #       R   8.3144598
        #       F   96485.33289
        #       nao 140   mM
        #       nai  10   mM
        #       ko    2.5 mM
        #       ki   64.4 nM
        # We don't use the default values for ena and ek taken
        # from the HH paper:
        #   ena    = 115.0mV + -65.0mV,
        #   ek     = -12.0mV + -65.0mV,
        ena    =  63.55148117386mV,
        ek     = -74.17164678272mV,

        c_m    = 0.01F*m^-2,
        gnabar = .12S*cm^-2,
        gkbar  = .036S*cm^-2,
        gl     = .0003S*cm^-2,
        el     = -54.3mV,
        q10    = 1
    ) = new(c_m, gnabar, gkbar, gl, ena, ek, el, q10)

end

struct Stim
    t0        # start time of stimulus
    t1        # stop time of stimulus
    i_e       # stimulus current density

    Stim() = new(0s, 0s, 0A/m^2)
    Stim(t0, t1, i_e) = new(t0, t1, i_e)
end

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

vtrap(x,y) = x/(exp(x/y) - 1.0)

# "m" sodium activation system
function m_lims(v, q10)
    alpha = .1mV^-1 * vtrap(-(v+40mV),10mV)
    beta =  4 * exp(-(v+65mV)/18mV)
    sum = alpha + beta
    mtau = 1ms / (q10*sum)
    minf = alpha/sum
    return mtau, minf
end

# "h" sodium inactivation system
function h_lims(v, q10)
    alpha = 0.07*exp(-(v+65mV)/20mV)
    beta = 1 / (exp(-(v+35mV)/10mV) + 1)
    sum = alpha + beta
    htau = 1ms / (q10*sum)
    hinf = alpha/sum
    return htau, hinf
end

# "n" potassium activation system
function n_lims(v, q10)
    alpha = .01mV^-1 * vtrap(-(v+55mV),10mV)
    beta = .125*exp(-(v+65mV)/80mV)
    sum = alpha + beta
    ntau = 1ms / (q10*sum)
    ninf = alpha/sum
    return ntau, ninf
end

# Choose initial conditions for the system such that the gating variables
# are at steady state for the user-specified voltage v
function initial_conditions(v, q10)
    mtau, minf = m_lims(v, q10)
    htau, hinf = h_lims(v, q10)
    ntau, ninf = n_lims(v, q10)

    return (v, minf, hinf, ninf)
end

# Given time t and state (v, m, h, n),
# return (vdot, mdot, hdot, ndot)
function f(t, state; p=HHParam(), stim=Stim())
    v, m, h, n = state

    # calculate current density due to ion channels
    gna = p.gnabar*m*m*m*h
    gk = p.gkbar*n*n*n*n


    ina = gna*(v - p.ena)
    ik = gk*(v - p.ek)
    il = p.gl*(v - p.el)

    itot = ik + ina + il

    # calculate current density due to stimulus
    if t>=stim.t0 && t<stim.t1
        itot -= stim.i_e
    end
        
    # calculate the voltage dependent rates for the gating variables
    mtau, minf = m_lims(v, p.q10)
    htau, hinf = h_lims(v, p.q10)
    ntau, ninf = n_lims(v, p.q10)

    return (-itot/p.c_m, (minf-m)/mtau, (hinf-h)/htau, (ninf-n)/ntau)
end

function run_hh(t_end; v0=-65mV, stim=Stim(), param=HHParam(), sample_dt=0.01ms)
    v_scale = 1V
    t_scale = 1s

    v0, m0, h0, n0 = initial_conditions(v0, param.q10)
    y0 = [ v0/v_scale, m0, h0, n0 ]

    samples = collect(0s: sample_dt: t_end)

    fbis(t, y, ydot) = begin
        vdot, mdot, hdot, ndot =
            f(t*t_scale, (y[1]*v_scale, y[2], y[3], y[4]), stim=stim, p=param)

        ydot[1], ydot[2], ydot[3], ydot[4] =
            vdot*t_scale/v_scale, mdot*t_scale, hdot*t_scale, ndot*t_scale

        return Sundials.CV_SUCCESS
    end

    # Ideally would run with vector absolute tolerance to account for v_scale,
    # but this would prevent us using the nice cvode wrapper.

    res = Sundials.cvode(fbis, y0, scale.(samples, t_scale), abstol=1e-6, reltol=5e-10)

    return samples, res[:, 1]*v_scale
end

end # module HHChannels
