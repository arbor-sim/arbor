NEURON {
    SUFFIX hh05
    USEION k READ ek WRITE ik
    NONSPECIFIC_CURRENT il
    : qt is ASSIGNED, but needs to be RANGE
    RANGE gl, el, gkbar, q10
}

ASSIGNED { q10 }

STATE { n }

PARAMETER {
    gkbar  =   0.036  (S/cm2)
    gl     =   0.0003 (S/cm2)
    el     = -54.3    (mV)
    celsius           (degC)
    v                 (mV)
}


INITIAL {
    LOCAL alpha, beta

    q10 = 3^(0.1*celsius - 0.63)

    : potassium activation system
    alpha = n_alpha(v)
    beta  = n_beta(v)
    n     = alpha/(alpha + beta)
}

DERIVATIVE states {
    LOCAL alpha, beta

    : potassium activation system
    alpha = n_alpha(v)
    beta  = n_beta(v)
    n'    = (alpha - n*(alpha + beta))*q10
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL gk, gna, n2

    gk = gkbar*n*n*n*n

    ik  = gk*(v - ek)
    il  = gl*(v - el)
}

FUNCTION m_alpha(v) { m_alpha = exprelr(-0.1*v - 4.0) }
FUNCTION h_alpha(v) { h_alpha = 0.07*exp(-0.05*v - 3.25) }
FUNCTION n_alpha(v) { n_alpha = 0.1*exprelr(-0.1*v - 5.5) }

FUNCTION m_beta(v)  { m_beta  = 4.0*exp(-(v + 65.0)/18.0) }
FUNCTION h_beta(v)  { h_beta  = 1.0/(exp(-0.1*v - 3.5) + 1.0) }
FUNCTION n_beta(v)  { n_beta  = 0.125*exp(-0.0125*v - 0.8125) }
