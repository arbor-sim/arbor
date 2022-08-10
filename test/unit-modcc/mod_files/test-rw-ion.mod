NEURON {
    SUFFIX hh
    USEION na READ ena, ina WRITE ina
}

BREAKPOINT { ina = 5*(v - ena) }

INITIAL {}
