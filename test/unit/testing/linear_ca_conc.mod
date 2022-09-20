: Test mechanism with linear response to ica.

NEURON {
    SUFFIX linear_ca_conc
    USEION ca READ ica WRITE cai VALENCE 2
    RANGE coeff
}

PARAMETER {
    coeff = 0
}

ASSIGNED {}

STATE {
    cai
}

INITIAL {
    cai = 0
}

BREAKPOINT  {
    SOLVE update METHOD cnexp
}

DERIVATIVE update {
    cai' = -coeff*ica
}

