NEURON {
    SUFFIX test0_kin_steadystate
}

STATE {
    s d h
}

BREAKPOINT {
    SOLVE state STEADYSTATE sparse
}

KINETIC state {
    LOCAL alpha1, beta1, alpha2, beta2
    alpha1 = 2
    beta1 = 0.6
    alpha2 = 3
    beta2 = 0.7

    ~ s <-> h (alpha1, beta1)
    ~ d <-> s (alpha2, beta2)

    CONSERVE s + d + h = 1
}

INITIAL {
    h = 0.2
    d = 0.3
    s = 1-d-h
}
