: Test mechanism generating a fixed ionic current

NEURON {
    SUFFIX fixed_ica_current
    USEION ca WRITE ica
    RANGE current_density
}

PARAMETER {
    current_density = 0
}

ASSIGNED {}

INITIAL {
    ica = current_density
}

STATE {}

BREAKPOINT  {
    ica = current_density
}

