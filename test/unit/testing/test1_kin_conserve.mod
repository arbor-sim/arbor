NEURON {
    SUFFIX test1_kin_conserve
}

STATE {
    a b x y
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    LOCAL alpha1, beta1, alpha2, beta2
    alpha1 = 2
    beta1 = 0.6
    alpha2 = 3
    beta2 = 0.7

    ~ a <-> b (alpha1, beta1)
    ~ x <-> y (alpha2, beta2)

    CONSERVE a + b = 1
    CONSERVE x + y = 1
}

INITIAL {
    a = 0.2
    b = 1 - a
    x = 0.6
    y = 1 - x
}
