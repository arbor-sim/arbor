NEURON {
    SUFFIX diam_test
}

PARAMETER { diam area }

STATE {
    d
    a
}

BREAKPOINT {
    SOLVE state
}

DERIVATIVE state {
    d = diam
    a = area
}

INITIAL {
    d = -23.0
    a = -42.0
}

