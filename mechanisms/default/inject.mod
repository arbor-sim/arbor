NEURON {
  POINT_PROCESS inject
  USEION na WRITE nad
}

BREAKPOINT {}

NET_RECEIVE(weight) {
    nad = nad + 200.0*weight
}
