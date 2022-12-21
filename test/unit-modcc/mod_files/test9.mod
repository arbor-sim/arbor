NEURON {
    POINT_PROCESS OrnsteinUhlenbeck
	GLOBAL tau, sigma
}

PARAMETER {
    tau = 1000
	sigma = 0.1
}

ASSIGNED {}

STATE {
    y
}

INITIAL {
    y=1
}

BREAKPOINT {
    SOLVE state METHOD stochastic
}

WHITE_NOISE {
    zeta zeta2
}

DERIVATIVE state {
    y' = -y/tau + (2*sigma*sigma/tau)^0.5*zeta
}
