NEURON {
    SUFFIX hh
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    NONSPECIFIC_CURRENT il
    RANGE gnabar, gkbar, gl, el
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gnabar = .12 (S/cm2)
    gkbar = .036 (S/cm2)
    gl = .0003 (S/cm2)
    el = -54.3 (mV)
    celsius
}

STATE {
    m h n
}

ASSIGNED {
    v (mV)
    q10
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL gna, gk, n2, n_, m_
    n_ = n
    n2 = n_*n_
    m_ = m
    gna = gnabar*m_*m_*m_*h
    ina = gna*(v - ena)
    gk = gkbar*n2*n2
    ik = gk*(v - ek)
    il = gl*(v - el)
}

INITIAL {
    q10 = 3^((celsius - 6.3)*0.1)

    LOCAL alpha, beta

    :"m" sodium activation system
    alpha = exprelr(-(v+40)*0.1)
    beta =  4 * exp(-(v+65)*0.05555555555555555)
    m = alpha/(alpha + beta)

    :"h" sodium inactivation system
    alpha = .07 * exp(-(v+65)*0.05)
    beta = 1 / (exp(-(v+35)*0.1) + 1)
    h = alpha/(alpha + beta)

    :"n" potassium activation system
    alpha = .1*exprelr(-(v+55)*0.1)
    beta = .125*exp(-(v+65)*0.0125)
    n = alpha/(alpha + beta)
}

DERIVATIVE states {
    LOCAL alpha, beta, sum, minf, ninf, hinf, mrate, nrate, hrate

    :"m" sodium activation system
    alpha = exprelr(-(v+40)*0.1)
    beta =  4 * exp(-(v+65)*0.05555555555555555)
    sum = alpha + beta
    mrate = q10*sum
    minf = alpha/sum

    :"h" sodium inactivation system
    alpha = .07 * exp(-(v+65)*0.05)
    beta = 1 / (exp(-(v+35)*0.1) + 1)
    sum = alpha + beta
    hrate = q10*sum
    hinf = alpha/sum

    :"n" potassium activation system
    alpha = .1*exprelr(-(v+55)*0.1)
    beta = .125*exp(-(v+65)*0.0125)
    sum = alpha + beta
    nrate = q10*sum
    ninf = alpha/sum

    m' = (minf-m)*mrate
    h' = (hinf-h)*hrate
    n' = (ninf-n)*nrate
}
