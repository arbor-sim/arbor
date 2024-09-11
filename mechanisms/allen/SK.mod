: SK-type calcium-activated potassium current
: Reference : Kohler et al. 1996

NEURON {
    SUFFIX SK
    USEION k READ ek WRITE ik
    USEION ca READ cai
    RANGE gbar
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

CONSTANT { eps = 1e-7  }

STATE { z }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL g
    g = gbar*z
    ik =  g*(v - ek)
}

DERIVATIVE states { z' = (z_inf(cai) - z) / zTau }

INITIAL { z = z_inf(cai) }

FUNCTION z_inf(cai) {
    if(cai < eps) {
         z_inf = 1/(1 + (0.00043 / (cai + eps))^4.8)
    } else {
         z_inf = 1/(1 + (0.00043 / cai)^4.8)
    }
}
