: Test density mechanism for passive calcium current.

NEURON {
    SUFFIX ca_linear
    USEION ca WRITE ica VALENCE 2
    RANGE g
}

PARAMETER {
    g = .001 (S/cm2)
}

ASSIGNED {}

INITIAL {}

BREAKPOINT  {
    ica = g*v
}
