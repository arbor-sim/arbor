NEURON {
    SUFFIX celsius_test
}

PARAMETER {
    celsius
}

STATE {
    c
}

ASSIGNED {
}

BREAKPOINT {
    SOLVE states
}

DERIVATIVE states {
    c = celsius
}

INITIAL {
    c = 0
}

