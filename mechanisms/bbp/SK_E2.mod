: SK-type calcium-activated potassium current
: Reference : Kohler et al. 1996

NEURON {
    SUFFIX SK_E2
    USEION k READ ek WRITE ik
    USEION ca READ cai
    RANGE gSK_E2bar
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (mM) = (milli/liter)
}

PARAMETER {
    gSK_E2bar = 0.000001 (mho/cm2)
    zTau      = 1        (ms)
}

STATE {
    z FROM 0 TO 1
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gSK_E2bar*z*(v - ek)
}

DERIVATIVE states {
    z' = (z_inf(cai) - z)/zTau
}

INITIAL {
    z = z_inf(cai)
}

FUNCTION z_inf(ca) {
    if (ca < 1e-7) {
        z_inf = 0
    } else {
        z_inf =  1/(1 + (0.00043/ca)^4.8)
    }
}
