NEURON {
  POINT_PROCESS inject
  USEION x WRITE xd, ix
  RANGE alpha, beta
}

ASSIGNED { beta }

PARAMETER { alpha = 200 }

INITIAL { beta = 0 }

BREAKPOINT {
    xd = xd + beta
    ix = ix + beta
    beta = 0
}

NET_RECEIVE(weight) {
    beta = alpha*weight
}
