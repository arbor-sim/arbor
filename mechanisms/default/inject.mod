NEURON {
  POINT_PROCESS inject
  USEION x WRITE xd
}

BREAKPOINT {}

NET_RECEIVE(weight) {
    xd = xd + 200.0*weight
}
