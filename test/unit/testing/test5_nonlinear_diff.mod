NEURON {
    SUFFIX test5_nonlinear_diff
}

STATE {
    a b c
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

DERIVATIVE state {
    LOCAL f0, f1, r0, r1
    f0 = 2
    r0 = 1
    f1 = 3
    r1 = 0

    a' = -f0*a*b + r0*c
    b' = -f0*a*b -r1*b + (r0+f1)*c
    c' =  f0*a*b +r1*b - (r0+f1)*c
}

INITIAL {
    a = 0.2
    b = 0.3
    c = 0.5
}
