: Reference: Kole, Hallermann, and Stuart
:            J. Neurosci. 2006

NEURON {
    SUFFIX Ih
    NONSPECIFIC_CURRENT ihcn
    RANGE gbar
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar =   0.00001 (S/cm2)
    ehcn = -45.0     (mV)
}

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL g
    g = gbar*m
    ihcn = g*(v - ehcn)
}

DERIVATIVE states {
    LOCAL ma, mb
    ma = m_alpha(v)
    mb = m_beta(v)
    m' = ma - m*(ma + mb)
}

INITIAL {
    LOCAL ma, mb
    ma = m_alpha(v)
    mb = m_beta(v)
    m  = ma/(ma + mb)
}

: Traps for 0 in denominator of rate equations
FUNCTION vtrap(x, y) { vtrap   = y*exprelr(x/y) }
FUNCTION m_alpha(v)  { m_alpha = 0.001 * 6.43 * vtrap(v + 154.9, 11.9) }
FUNCTION m_beta(v)   { m_beta  = 0.001*193*exp(v/33.1) }
