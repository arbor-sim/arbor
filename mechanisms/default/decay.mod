NEURON {
    SUFFIX decay
    USEION x WRITE xd
    RANGE F, tau
}

PARAMETER { tau = 5 }

INITIAL { F = xd }

STATE { F }

BREAKPOINT {
   SOLVE dF METHOD cnexp
   xd = F
}

DERIVATIVE dF {
   F = xd
   F' = -tau*F
}
