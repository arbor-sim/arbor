NEURON {
    SUFFIX decay
    USEION x WRITE xd, ix
}

INITIAL { F = xd }

STATE { F }

BREAKPOINT {
   SOLVE dF METHOD cnexp
   xd = F
}

DERIVATIVE dF {
   F = xd
   F' = -5*F
}
