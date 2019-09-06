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
    f = 10
    r = 10
}

STATE {
    a (mA/cm2)
    b (mA/cm2)
    c
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
    ~ 2a  + b <-> c (f, r)
}

