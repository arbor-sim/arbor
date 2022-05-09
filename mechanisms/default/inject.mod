NEURON {
  POINT_PROCESS inject
  USEION x WRITE xd
  RANGE alpha
}

PARAMETER {
    alpha = 200
}

BREAKPOINT {}

NET_RECEIVE(weight) {
    xd = xd + alpha*weight
}
