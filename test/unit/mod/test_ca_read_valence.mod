: Test mechanism for checking ionic valence read

NEURON {
    SUFFIX test_ca_read_valence
    USEION ca READ ica VALENCE zca
}

PARAMETER {}

ASSIGNED {}

STATE {
    record_zca
}

INITIAL {
    record_zca = zca
}

BREAKPOINT  {
}

