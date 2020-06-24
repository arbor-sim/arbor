: Comment: Kv3-like potassium current

NEURON {
    SUFFIX Kv3_1
    USEION k READ ek WRITE ik
    RANGE gbar, ik 
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar   = 0.00001 (S/cm2)
    vshift = 0       (mV)
}

STATE { 
    m
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar*m*(v - ek)
}

DERIVATIVE states {
    LOCAL mInf, mRat
    mInf =    1/(1 + exp(-(v - 18.700 - vshift)/9.700))
    mRat = 0.25*(1 + exp(-(v + 46.560 - vshift)/44.140))
    
    m' = (mInf - m)*mRat
}

INITIAL {
    m = 1/(1 + exp(-(v - 18.700 - vshift)/9.700))
}
