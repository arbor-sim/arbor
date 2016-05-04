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
    e = -65 (mV) : we use -65 for the ball and stick model, instead of Neuron default of -70
}

ASSIGNED {
    v (mV)
}

BREAKPOINT {
    i = g*(v - e)
}
