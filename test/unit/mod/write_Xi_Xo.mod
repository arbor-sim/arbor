: Ion concentration writer, with state variable.
: Used for testing mechanism and ion probes.

NEURON {
    SUFFIX write_Xi_Xo
    USEION x WRITE xi, xo
    GLOBAL xi0, xo0, s0
}

PARAMETER {
    xi0 = 1
    xo0 = 2
    s0 = 3
}

ASSIGNED {}

STATE {
    s
}

INITIAL {
    s = s0
    xi = xi0
    xo = xo0
}

BREAKPOINT  {
    s = s0
    xi = xi0
    xo = xo0
}

