: Stochastic Volatility (linear SDE)
: ===================================
: P:     log of the price (of a stock or bond)
: sigma: volatility
: The observed volatility of the price is not constant but itself a stochastic process.

NEURON {
    POINT_PROCESS stochastic_volatility
    GLOBAL mu, theta, kappa, sigma_1
}

PARAMETER {
    mu      = 0.1
    theta   = 0.1
    kappa   = 0.1
    sigma_1 = 0.1
}

ASSIGNED {}

STATE {
    P sigma
}

INITIAL {
    P=1
    sigma=0.2
}

BREAKPOINT {
    SOLVE state METHOD stochastic
}

WHITE_NOISE {
    W1 W2
}

DERIVATIVE state {
    P'     = mu + sigma*W1
    sigma' = kappa*(theta-sigma) + sigma_1*W2
}

