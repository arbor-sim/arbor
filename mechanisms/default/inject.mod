NEURON {
  POINT_PROCESS inject
  USEION na WRITE nad
}


PARAMETER {
  alpha = 200
}

BREAKPOINT {}

NET_RECEIVE(weight) {
    nad = nad + alpha*weight
}
