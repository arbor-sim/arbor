NEURON {
  POINT_PROCESS inject_norm_amount
  USEION x WRITE xd
  RANGE alpha, beta
}

ASSIGNED {
    beta
}

PARAMETER {
	alpha = 200
	area
	diam
}

INITIAL { 
	beta = 0
}

BREAKPOINT {
    xd = xd + beta * area / 1000 : considering particle amount -> normalization by area
    beta = 0
}

NET_RECEIVE(weight) {
    beta = alpha*weight
}
