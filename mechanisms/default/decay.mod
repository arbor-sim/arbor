NEURON {
    SUFFIX decay
    USEION x WRITE xd
    RANGE tau
}

PARAMETER { tau = 5 }

BREAKPOINT {
   SOLVE dX METHOD cnexp
}

DERIVATIVE dX {
   xd' = -tau*xd
}
