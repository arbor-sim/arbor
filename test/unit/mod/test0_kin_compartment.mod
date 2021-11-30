NEURON {
    SUFFIX test0_kin_compartment
}

STATE {
    s d h
}

PARAMETER {
    A = 0.5
    B = 0.1
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    COMPARTMENT A {s h}
    COMPARTMENT B {d}

    LOCAL alpha1, beta1, alpha2, beta2
    alpha1 = 2
    beta1 = 0.6
    alpha2 = 3
    beta2 = 0.7

    ~ s <-> h (alpha1, beta1)
    ~ d <-> s (alpha2, beta2)
}

INITIAL {
    h = 0.2
    d = 0.3
    s = 1-d-h
}
