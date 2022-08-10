NEURON {
    JUNCTION_PROCESS gj
    NONSPECIFIC_CURRENT i
    RANGE g
}

INITIAL {}

PARAMETER { g = 1 }

BREAKPOINT {
    i = g*(v - v_peer)
}
