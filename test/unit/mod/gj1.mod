NEURON {
    JUNCTION_PROCESS gj1
    NONSPECIFIC_CURRENT i
    RANGE g, e
}
INITIAL {}

PARAMETER {
    g = 1
    e = 0
}

BREAKPOINT {
    i = g*(v - v_peer - e)
}
