: Test mechanism with linear response to ica.

NEURON {
    SUFFIX read_cai_init
    USEION ca READ cai VALENCE 2
}

PARAMETER {}

ASSIGNED {}

STATE {
    s
}

INITIAL {
    s = cai
}

BREAKPOINT {
    SOLVE states
}

DERIVATIVE states {
    s = cai
}

