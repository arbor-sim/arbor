NEURON {
    SUFFIX hh03
    NONSPECIFIC_CURRENT il
}

PARAMETER {
    gl =   0.0003 (S/cm2)
    el = -54.3    (mV)
}

BREAKPOINT {
    il = gl*(v - el)
}
