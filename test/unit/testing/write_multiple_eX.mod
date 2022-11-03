: Reversal potential writer that writes to two different ions
: as a simple function of internal concentration.

NEURON {
    SUFFIX write_multiple_eX
    USEION x READ xi WRITE ex
    USEION y READ yi WRITE ey
}

PARAMETER {}

ASSIGNED {}

STATE {}

INITIAL {
    ex = 100 + xi
    ey = 200 + yi
}

BREAKPOINT  {
    ex = 100 + xi
    ey = 200 + yi
}

