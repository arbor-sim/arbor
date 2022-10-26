: Reversal potential writer that writes to a single ion
: as a simple function of internal concentration.

NEURON {
    SUFFIX write_eX
    USEION x READ xi WRITE ex
}

PARAMETER {}

ASSIGNED {}

STATE {}

INITIAL {
    ex = 100 + xi
}

BREAKPOINT  {
    ex = 100 + xi
}

