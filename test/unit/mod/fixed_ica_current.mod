: Test mechanism generating a fixed ionic current

NEURON {
    SUFFIX fixed_ica_current
    USEION ca WRITE ica VALENCE 2
    RANGE ica_density
}

PARAMETER {
    ica_density = 0
}

ASSIGNED {}

INITIAL {
    ica = ica_density
}

STATE {}

BREAKPOINT  {
    ica = ica_density
}

