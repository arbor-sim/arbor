NEURON {
  POINT_PROCESS inject
  USEION na WRITE nad
}


PARAMETER {
  alpha = 2.0
}

STATE { nad }

INITIAL {
     nad = 2.0
}

BREAKPOINT {}

NET_RECEIVE(weight) {
    nad = nad + alpha*weight
}
