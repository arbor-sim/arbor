: Input current modeled by an Ornstein-Uhlenbeck process
: Sets non-specific current given mean, volatility and relaxation time
:
: Adapted from Jannik Luboeinski's example here: https://github.com/jlubo/arbor_ou_lif_example
:
: I_ou  input current (nA)
: μ     input current mean (nA)
: σ     input current volatility (nA)
: τ     relaxation time (ms)
:
: dI_ou/dt = -(1/τ) * (I_ou - μ) + √(2/τ) * σ * W
:
: statistic properties:
:   E[I_ou] = μ - (μ - I_ou_0) * e^(-t/τ)
: Var[I_ou] = σ^2 * (1 - e^(-2 * t/τ))
:
: in the limit t->∞:
:   E[I_ou] = μ
: Var[I_ou] = σ^2

NEURON {
    POINT_PROCESS ou_input
    RANGE mu, sigma, tau
    NONSPECIFIC_CURRENT I
}

UNITS {
    (ms) = (milliseconds)
    (nA) = (nanoampere)
}

PARAMETER {
    mu    = 1 (nA) : mean of the stochastic process
    sigma = 1 (nA) : volatility of the stochastic process
    tau   = 1 (ms) : relaxation time
}

STATE {
    I_ou (nA) : instantaneous state
    active    : indicates if the process is currently enabled 
}

ASSIGNED {
    alpha
    beta
}

WHITE_NOISE {
    W
}

INITIAL {
    I_ou   = 0
    active = -1
    alpha  = 1.0/tau
    beta   = sigma * sqrt(2.0/tau)
}

BREAKPOINT {
    SOLVE state METHOD stochastic
    I = -I_ou
}

DERIVATIVE state {
    I_ou' = step_right(active) * (alpha * (mu - I_ou) + beta * W)
}

NET_RECEIVE(weight) {
    if (weight >= 0) { : indicates that stimulation begins
        I_ou = mu : initialize the process at the mean value
        active = 1
    }
    else { : indicates that stimulation ends
        I_ou = 0 : switch off the process
        active = -1
    }
}
