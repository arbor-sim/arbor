: Copy range parameter to state variable exactly.

NEURON {
    SUFFIX param_as_state
    RANGE p
}

PARAMETER {
    p = 1
}

ASSIGNED {}

STATE {
    s
}

INITIAL {
    s = p
}

BREAKPOINT  {
}

