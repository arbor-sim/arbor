NEURON {
    SUFFIX test1_kin_diff
}

STATE {
    a b x y
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    LOCAL alpha, beta, gamma, delta
    alpha = 2
    beta = 0.6
    gamma = 3
    delta = 0.7

    ~ a <-> b (alpha, beta)
    ~ x <-> y (gamma, delta)
}

INITIAL {
    a = 0.2
    b = 1 - a
    x = 0.6
    y = 1 - x
}
