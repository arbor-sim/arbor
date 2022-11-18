: Ornstein-Uhlenbeck: linear mean reverting process
: =================================================
: dS(t) = kappa [mu - S(t)] dt + sigma dW(t)
:     E[S(t)] = mu - (mu-S_0)*e^(-kappa t)
:   Var[S(t)] = sigma^2/(2 kappa) [1 - e^(-2 kappa t)]
: in the limit:
:     lim t->∞   E[S(t)] = mu
:     lim t->∞ Var[S(t)] = sigma^2/(2 kappa)

NEURON {
    SUFFIX ornstein_uhlenbeck
    GLOBAL kappa, mu, sigma
}

PARAMETER {
    kappa = 0.1
    mu = 1
    sigma = 0.1
}

STATE {
    S
}

INITIAL {
    S=2
}

BREAKPOINT {
    SOLVE state METHOD stochastic
}

WHITE_NOISE { W }

DERIVATIVE state {
    S' = kappa*(mu - S) + sigma*W
}

