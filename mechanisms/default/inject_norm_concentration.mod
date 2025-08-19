NEURON {
  POINT_PROCESS inject_norm_concentration
  USEION x WRITE xd
  RANGE alpha, beta
}

ASSIGNED {
    beta
    volume
}

PARAMETER {
	alpha = 200
	area
	diam
}

INITIAL { 
	beta = 0
	volume = area*diam/4 : = pi*r^2*h
}

BREAKPOINT {
    xd = xd + beta * area / 1000 / volume : considering particle concentration -> normalization by area and volume
    beta = 0
}

NET_RECEIVE(weight) {
    beta = alpha*weight
}
