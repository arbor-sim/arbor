NEURON {
    POINT_PROCESS mean_reverting_stochastic_process2
    GLOBAL kappa, mu, sigma
}

PARAMETER {
    kappa = 0.1
    mu = 1
    sigma = 0.1
}

ASSIGNED {}

STATE {
    X
    Y
}

INITIAL {
    X=2
    Y=1
}

BREAKPOINT {
    SOLVE state METHOD stochastic
}

WHITE_NOISE {
    Q Z
}

DERIVATIVE state {
    X' = kappa*(mu - X) + sigma*Q
    Y' = kappa*(mu - Y) + sigma*Z
}

