NEURON {
    JUNCTION_PROCESS gj0
    NONSPECIFIC_CURRENT i
    RANGE g
}
INITIAL {}

PARAMETER {
    g = 1
}

BREAKPOINT {
    i = g*(v - v_peer)
}
