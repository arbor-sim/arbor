NEURON {
    SUFFIX mean_reverting_stochastic_density_process
    GLOBAL kappa, mu, sigma
}

PARAMETER {
    kappa = 0.1
    mu = 1
    sigma = 0.1
}

STATE { s }

WHITE_NOISE { w }

BREAKPOINT {
    SOLVE states METHOD stochastic
}

INITIAL {
    s = 2
}

DERIVATIVE states {
    s' = kappa*(mu - s) + sigma*w
}
