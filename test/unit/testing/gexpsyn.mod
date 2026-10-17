NEURON {
    POINT_PROCESS gexpsyn
    RANGE tau
    GLOBAL e
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau = 2.0 (ms) : the default for Neuron is 0.1
    e = 0   (mV)
}

ASSIGNED {}

STATE {
    g (uS)
}

INITIAL {
    g = 0
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
