NEURON {
    SUFFIX test_kin1
    NONSPECIFIC_CURRENT il
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}

PARAMETER {
    tau = 10 (ms)
}

STATE {
    a (mA/cm2)
    b (mA/cm2)
}

ASSIGNED {
    v (mV)
}

BREAKPOINT {
    SOLVE states METHOD sparse
    il = 0*v+a
}

INITIAL {
    a = 0.01
    b = 0
}

KINETIC states {
    ~ a <-> b (2/3/tau, 1/3/tau)
}

