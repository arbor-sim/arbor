NEURON {
    SUFFIX mean_reverting_stochastic_density_process2
	GLOBAL kappa, mu, sigma
}

PARAMETER {
    kappa = 0.1
    mu = 1
	sigma = 0.1
}

STATE { x y }

WHITE_NOISE { q b }

BREAKPOINT {
    SOLVE states METHOD stochastic
}

INITIAL {
    x = 2
    y = 1
}

DERIVATIVE states {
    x' = kappa*(mu - x) + sigma*q
    y' = kappa*(mu - y) + sigma*b
}

