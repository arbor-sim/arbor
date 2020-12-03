NEURON {
    POINT_PROCESS post_events_syn
    RANGE tau, e
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    tau = 2.0 (ms) : the default for Neuron is 0.1
    e = 0   (mV)
}

STATE {
    g
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

POST_EVENT(t) {
   g = g + (0.1*t)
}
