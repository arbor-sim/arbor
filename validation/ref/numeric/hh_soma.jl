using Sundials
using SIUnits.ShortUnits

c_m = 0.01nF*m^-2

radius = 18.8μm / 2
surface_area = 4 * pi * radius * radius

gnabar = .12S*cm^-2
gkbar  = .036S*cm^-2
gl     = .0003S*cm^-2
el     = -54.3mV
#celsius= 6.3
#q10 = 3^((celsius - 6.3)/10)
q10 = 1

# define the resting potential for the membrane
vrest = -65.0mV

# define the resting potentials for ion species
ena = 115.0mV+vrest
ek  = -12.0mV+vrest
eca =  12.5mV*log(2.0/5e-5)

vtrap(x,y) = x/(exp(x/y) - 1.0)

function print()
    println("q10 ", q10)
    println("vrest ", vrest)
    println("ena ", ena)
    println("ek ", ek)
    println("eca ", eca)
end

#"m" sodium activation system
function m_lims(v)
    alpha = .1mV^-1 * vtrap(-(v+40mV),10mV)
    beta =  4 * exp(-(v+65mV)/18mV)
    sum = alpha + beta
    mtau = 1ms / (q10*sum)
    minf = alpha/sum
    return mtau, minf
end

#"h" sodium inactivation system
function h_lims(v)
    alpha = 0.07*exp(-(v+65mV)/20mV)
    beta = 1 / (exp(-(v+35mV)/10mV) + 1)
    sum = alpha + beta
    htau = 1ms / (q10*sum)
    hinf = alpha/sum
    return htau, hinf
end

#"n" potassium activation system
function n_lims(v)
    alpha = .01mV^-1 * vtrap(-(v+55mV),10mV)
    beta = .125*exp(-(v+65mV)/80mV)
    sum = alpha + beta
    ntau = 1ms / (q10*sum)
    ninf = alpha/sum
    return ntau, ninf
end

# v = y[1] V
# m = y[2]
# h = y[3]
# n = y[4]

# dv/dt = ydot[1] V/s
# dm/dt = ydot[2] /s
# dh/dt = ydot[3] /s
# dn/dt = ydot[4] /s

# choose initial conditions for the system such that the gating variables
# are at steady state for the user-specified voltage v
function initial_conditions(v)
    mtau, minf = m_lims(v)
    htau, hinf = h_lims(v)
    ntau, ninf = n_lims(v)

    return [v/V, minf, hinf, ninf]
end

# calculate the lhs of the ODE system
function f(t, y, ydot)
    # copy variables into helper variable
    v = y[1]V
    m, h, n = y[2], y[3], y[4]

    # calculate current due to ion channels
    gna = gnabar*m*m*m*h
    gk = gkbar*n*n*n*n
    ina = gna*(v - ena)
    ik = gk*(v - ek)
    il = gl*(v - el)
    imembrane = ik + ina + il

    # calculate current due to stimulus
    #c.add_stimulus({0,0.5}, {10., 100., 0.1});
    ielectrode = 0.0nA / surface_area
    time = t*ms
    if time>=10ms && time<100ms
        ielectrode = 0.1nA / surface_area
    end

    # calculate the total membrane current
    i = -imembrane + ielectrode

    # calculate the voltage dependent rates for the gating variables
    mtau, minf = m_lims(v)
    ntau, ninf = n_lims(v)
    htau, hinf = h_lims(v)

    # set the derivatives
    # note values are in SI units, as determined by the scaling factors:
    ydot[1] = i/c_m         / (V/s)
    ydot[2] = (minf-m)/mtau / (1/s)
    ydot[3] = (hinf-h)/htau / (1/s)
    ydot[4] = (ninf-n)/ntau / (1/s)

    return Sundials.CV_SUCCESS
end

###########################################################
#   now we actually run the model
###########################################################

# from 0 to 100 ms in 1000 steps
t = collect(linspace(0.0, 0.1, 1001));

# these tolerances are as tight as they will go without breaking convergence of the iterative schemes
y0 = initial_conditions(vrest)
res = Sundials.cvode(f, y0, t, abstol=1e-6, reltol=5e-10);

#using Plots
#gr()
#plot(t, res[:,1])
