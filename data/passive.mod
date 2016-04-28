TITLE passive membrane channel

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}

NEURON {
    SUFFIX pas
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

INITIAL {}

PARAMETER {
    g = .001    (S/cm2) :<0,1e9>
    e = -70 (mV)
}

ASSIGNED {
    v (mV)
}

BREAKPOINT {
    i = g*(v - e)
}
