: Test mechanism for consuming ionic reversal potential

NEURON {
    SUFFIX read_eX
    USEION x READ ex
}

PARAMETER {}

ASSIGNED {}

STATE {
    record_ex
}

INITIAL {
    record_ex = ex
}

BREAKPOINT  {
    record_ex = ex
}

