NEURON {
    SUFFIX test_linear_state
    RANGE a0, a1, a3, a3
}

STATE {
    s d h
}

PARAMETER {
    a4 = 7.3
}

ASSIGNED { a0 a1 a2 a3 }

INITIAL {
    a0 = 2.5
    a1 = 0.5
    a2 = 3
    a3 = 2.3
}

BREAKPOINT {
    SOLVE sinit
}

LINEAR sinit {
    ~                (a4 - a3)*d -    a2*h = 0
    ~ (a0 + a1)*s - (-a1 + a0)*d           = 0
    ~           s +            d +       h = 1
}
