: Test point mechanism generating a fixed ionic current

NEURON {
    POINT_PROCESS point_ica_current
    USEION ca WRITE ica VALENCE 2
}

ASSIGNED {}

STATE {
    ica_nA
}

INITIAL {
    ica_nA = 0
}

BREAKPOINT  {
    ica = ica_nA
}

NET_RECEIVE(weight) {
    ica_nA = weight
}
