NEURON {
    SUFFIX SABR
	GLOBAL alpha, beta
}

PARAMETER {
    alpha = 1
	beta = 0.1
}

ASSIGNED {}

STATE {
    F
    sigma
}

INITIAL {
    F=1
    sigma=0.2
}

BREAKPOINT {
    SOLVE state METHOD stochastic
}

WHITE_NOISE {
    W Z
}

DERIVATIVE state {
    F' = sigma*F^beta*W
    sigma' = alpha*sigma*Z
}

