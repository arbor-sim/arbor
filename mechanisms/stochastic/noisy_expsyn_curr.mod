: Exponential current-based synapse
: Input current modeled by an Ornstein-Uhlenbeck process
: Sets non-specific current given mean, volatility and relaxation time
:
: I_ou  input current (nA)
: μ     input current mean (nA)
: σ     input current volatility (nA)
: τ     relaxation time (ms)
:
: dI_ou/dt = -(1/τ) * (I_ou - μ) + √(1/τ) * σ * W
:
: statistic properties:
:   E[I_ou] = μ - (μ - I_ou_0) * e^(-t/τ)
: Var[I_ou] = (σ^2 / 2) * (1 - e^(-2 * t/τ))
:
: in the limit t->∞:
:   E[I_ou] = μ
: Var[I_ou] = (σ^2 / 2)

NEURON {
    POINT_PROCESS noisy_expsyn_curr
    RANGE mu, sigma, tau
    NONSPECIFIC_CURRENT I
}

UNITS {
    (ms) = (milliseconds)
    (nA) = (nanoampere)
}

PARAMETER {
    mu    = 1 (nA)   : mean of the stochastic process
    sigma = 1 (nA)   : volatility of the stochastic process
    tau   = 5.0 (ms) : relaxation time
}

STATE {
    I_ou (nA)        : instantaneous state
    active           : indicates if the process is currently enabled 
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
    beta   = sigma * (1.0/tau)^0.5
}

BREAKPOINT {
    SOLVE state METHOD stochastic
    I = -I_ou
}

DERIVATIVE state {
    I_ou' = heaviside(active) * (alpha * (mu - I_ou) + beta * W)
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

FUNCTION heaviside(x) {
    if (x >= 0) {
        heaviside = 1
    }
    else {
        heaviside = 0
    }
}
