NEURON {
    SUFFIX test3_kin_diff
}

STATE {
    a b c
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    LOCAL f0, f1, r0, r1
    f0 = 2
    r0 = 1
    f1 = 3
    r1 = 0

    ~ a  + b <-> c (f0, r0)
    ~ c      <-> b (f1, r1)
}

INITIAL {
    a = 0.2
    b = 0.3
    c = 0.5
}
