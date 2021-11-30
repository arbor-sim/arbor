NEURON {
    SUFFIX test1_kin_steadystate
}

STATE {
    a b x y
}

PARAMETER {
    A = 0.5
    B = 0.1
}

BREAKPOINT {
    SOLVE state STEADYSTATE sparse
}

KINETIC state {
    COMPARTMENT A {a b}

    LOCAL alpha1, beta1, alpha2, beta2
    alpha1 = 2
    beta1 = 0.6
    alpha2 = 3
    beta2 = 0.7

    ~ a <-> b (alpha1, beta1)
    ~ x <-> y (alpha2, beta2)

    CONSERVE a + b = A
    CONSERVE x + y = 1
}

INITIAL {
    a = 0.2
    b = 1 - a
    x = 0.6
    y = 1 - x
}
