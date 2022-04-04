NEURON {
    SUFFIX decay
    USEION na WRITE nad
}

INITIAL { F = nad }

STATE { F }

BREAKPOINT {
   SOLVE dF METHOD cnexp
   nad = F
}

DERIVATIVE dF {
   F = nad
   F' = -5*F
}
