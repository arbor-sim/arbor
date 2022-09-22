NEURON {
    SUFFIX diam_test
}

PARAMETER {
    diam
}

STATE {
    d
}

ASSIGNED {
}

BREAKPOINT {
    SOLVE states
}

DERIVATIVE states {
    d = diam
}

INITIAL {
    d = 0
}

