: Reference: M-currents and other potassium currents in bullfrog sympathetic neurones
:            Adams et al. 1982
: Comment:   corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON {
    SUFFIX Im
    USEION k READ ek WRITE ik
    RANGE gbar, qt
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0.00001 (S/cm2) 
    v              (mV)
    celsius        (degC)
}

ASSIGNED { qt }

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL g
    g = gbar*m
    ik = g*(v - ek)
}

DERIVATIVE states {
    LOCAL a, b
    a = alpha(v)
    b = beta(v)
    m' = (a - m*(a + b))*qt
}

INITIAL {
    LOCAL a, b
    qt = 2.3^(0.1*(celsius - 21))
    a = alpha(v)
    b = beta(v)
    m = a/(a + b)
}

FUNCTION alpha(v) { alpha = 0.0033*exp( 0.1*(v + 35)) }
FUNCTION beta(v)  { beta  = 0.0033*exp(-0.1*(v + 35)) }
