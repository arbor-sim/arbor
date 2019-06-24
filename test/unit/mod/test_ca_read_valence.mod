: Test mechanism for checking ionic valence read

NEURON {
    SUFFIX test_ca_read_valence
    USEION ca READ ica VALENCE zca
}

PARAMETER {}

ASSIGNED {}

STATE {
    record_z
}

INITIAL {
    record_z = zca
}

BREAKPOINT  {
}

