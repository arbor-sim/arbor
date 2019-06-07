: Test mechanism for verifying ionic valence

NEURON {
    SUFFIX test_cl_valence
    USEION cl WRITE icl VALENCE -2
}

PARAMETER {}

ASSIGNED {}

INITIAL {}

STATE {}

BREAKPOINT  {
    icl = 1.23*(v-4.56)
}

