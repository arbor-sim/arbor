NEURON {
    SUFFIX test6_nonlinear_diff
}

STATE {
    p
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

DERIVATIVE state {
    p' = sin(p)
}

INITIAL {
    p = 1
}
