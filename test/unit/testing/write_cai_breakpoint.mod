: Test mechanism with linear response to ica.

NEURON {
    SUFFIX write_cai_breakpoint
    USEION ca WRITE cai READ ica VALENCE 2
}

PARAMETER {}

ASSIGNED {}

STATE {
    cai
}

INITIAL {}

BREAKPOINT {
    SOLVE states
}

DERIVATIVE states {
    cai = 5.2e-4
}