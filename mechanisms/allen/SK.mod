: SK-type calcium-activated potassium current
: Reference : Kohler et al. 1996

NEURON {
    SUFFIX SK
    USEION k READ ek WRITE ik
    USEION ca READ cai
    RANGE gbar, ik
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (mM) = (milli/liter)
}

PARAMETER {
    gbar = .000001 (mho/cm2)
    zTau = 1       (ms)
}

ASSIGNED {
    zInf
    g     (S/cm2)
}

STATE {
    z FROM 0 TO 1
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik   =  gbar*z*(v - ek)
}

DERIVATIVE states {
    LOCAL l_ca
    l_ca = cai
    if(l_ca < 1e-7){
        l_ca = l_ca + 1e-07
    }
    zInf = 1/(1 + (0.00043 / l_ca)^4.8)
    
    z' = (zInf - z) / zTau
}

INITIAL {
    LOCAL l_ca
    l_ca = cai
    if(l_ca < 1e-7) {
      l_ca = l_ca + 1e-07
    }
    
    z = 1/(1 + (0.00043 / l_ca)^4.8)
}
