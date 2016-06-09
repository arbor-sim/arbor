NEURON {
    POINT_PROCESS ExpSyn
    RANGE tau, e
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    : the default for Neuron is 0.1
    :tau = 0.1 (ms) : <1e-9,1e9>
    tau = 2.0 (ms) : <1e-9,1e9>
    e = 0   (mV)
}

ASSIGNED {}

STATE {
    g : (uS)
}

INITIAL {
    g=0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = g*(v - e)
}

DERIVATIVE state {
    g' = -g/tau
}

NET_RECEIVE(weight) {
    g = g + weight
}
