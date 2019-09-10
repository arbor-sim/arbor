NEURON {
    SUFFIX test1_kin_compartment
}

STATE {
    s d h
}

PARAMETER {
    A = 0.5
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    COMPARTMENT A {s h d}

    LOCAL alpha1, beta1, alpha2, beta2
    alpha1 = 2
    beta1 = 0.6
    alpha2 = 3
    beta2 = 0.7

    ~ s <-> h (alpha1, beta1)
    ~ d <-> s (alpha2, beta2)

    CONSERVE s + d + h = A
}

INITIAL {
    h = 0.2
    d = 0.3
    s = 1-d-h
}
