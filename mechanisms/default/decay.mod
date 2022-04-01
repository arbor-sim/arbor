NEURON {
    SUFFIX decay
    USEION na READ nai WRITE nad
    RANGE tau
}

STATE { nad }

INITIAL {
   nad = 2
}

PARAMETER {
   tau = 0.9
}

BREAKPOINT {
   SOLVE dX METHOD cnexp
}

DERIVATIVE dX {
   nad' = -tau*nad
}
