NEURON {
  POINT_PROCESS inject
  USEION x WRITE xd
  RANGE alpha, beta
}

ASSIGNED { beta }

PARAMETER { alpha = 200 }

INITIAL { beta = 0 }

BREAKPOINT {
    xd = xd + beta
    beta = 0
}

NET_RECEIVE(weight) {
    beta = alpha*weight
}
