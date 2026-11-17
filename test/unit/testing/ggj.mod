NEURON {
    JUNCTION_PROCESS ggj
    NONSPECIFIC_CURRENT i
    GLOBAL g
}

INITIAL {}

PARAMETER { g = 1 }

BREAKPOINT {
    i = g*(v - v_peer)
}
