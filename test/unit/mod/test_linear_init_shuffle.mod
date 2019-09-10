NEURON {
    SUFFIX test_linear_init_shuffle
    RANGE a0, a1, a3, a3
}

STATE {
    s d h
}

PARAMETER {
    a4 = 7.3
}

ASSIGNED {
    a0 : = 2.5
    a1 : = 0.5
    a2 : = 3
    a3 : = 2.3
}

BREAKPOINT {
    s = a1
}

INITIAL {
    SOLVE sinit
}

LINEAR sinit {
    ~ a4*d - a3*d - a2*h = 0
    ~ a0*s - a0*d = - a1*s - a1*d
    ~ s + d + h  = 1
}
