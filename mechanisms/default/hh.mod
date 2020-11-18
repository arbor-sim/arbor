NEURON {
    SUFFIX hh
    USEION na READ ena WRITE ina
    USEION k  READ ek  WRITE ik
    NONSPECIFIC_CURRENT il
    RANGE gnabar, gkbar, gl, el, q10
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gnabar =   0.12   (S/cm2)
    gkbar  =   0.036  (S/cm2)
    gl     =   0.0003 (S/cm2)
    el     = -54.3    (mV)
    celsius           (degC)
}

STATE { m h n }

ASSIGNED { q10 }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL gk, m_, n_, n2

    n_ = n
    m_ = m
    n2 = n_*n_
    gk = gkbar*n2*n2
    ina = gnabar*m_*m_*m_*h*(v - ena)
    ik  = gk*(v - ek)
    il  = gl*(v - el)
}

INITIAL {
    LOCAL alpha, beta

    q10 = 3^((celsius - 6.3)/10.0)
                            
    : sodium activation system
    alpha = m_alpha(v)
    beta  = m_beta(v)
    m     = alpha/(alpha + beta)

    : sodium inactivation system
    alpha = h_alpha(v)
    beta  = h_beta(v)
    h     = alpha/(alpha + beta)

    : potassium activation system
    alpha = n_alpha(v)
    beta  = n_beta(v)
    n     = alpha/(alpha + beta)
}

DERIVATIVE states {
    LOCAL alpha, beta, sum

    : sodium activation system
    alpha = m_alpha(v)
    beta  = m_beta(v)         
    sum   = alpha + beta
    m'    = (alpha - m*sum)*q10
                     
    : sodium inactivation system
    alpha = h_alpha(v)
    beta  = h_beta(v)
    sum   = alpha + beta
    h'    = (alpha - h*sum)*q10
                      
    : potassium activation system
    alpha = n_alpha(v)
    beta  = n_beta(v)                 
    sum   = alpha + beta
    n'    = (alpha - n*sum)*q10
}

FUNCTION vtrap(x,y) { vtrap   = y*exprelr(x/y) }

FUNCTION m_alpha(v) { m_alpha = 0.1*vtrap(-(v + 40.0), 10.0) }
FUNCTION h_alpha(v) { h_alpha = 0.07*exp(-(v + 65.0)/20.0) }
FUNCTION n_alpha(v) { n_alpha = 0.01*vtrap(-(v + 55.0), 10.0) }

FUNCTION m_beta(v)  { m_beta  = 4.0*exp(-(v + 65.0)/18.0) }
FUNCTION h_beta(v)  { h_beta  = 1.0/(exp(-(v + 35.0)/10.0) + 1.0) }
FUNCTION n_beta(v)  { n_beta  = 0.125*exp(-(v + 65.0)/80.0) }
