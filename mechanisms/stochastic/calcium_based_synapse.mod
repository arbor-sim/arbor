: Calcium-based plasticity model
: Based on the work of Graupner and Brunel, PNAS 109 (10): 3991-3996 (2012)
: https://doi.org/10.1073/pnas.1109359109, https://www.pnas.org/doi/10.1073/pnas.1220044110
:
: Author: Sebastian Schmitt
:
: The synapse is modeled as synaptic efficacy variable, ρ, which is a function of the calcium
: concentration, c(t). The synapse model features two stable states at ρ=0 (DOWN) and ρ=1 (UP),
: while ρ=ρ_star=0.5 represents a third unstable state between the two stable states.
: The calcium concentration dynamics are represented by a simplified model which ueses a linear sum
: of individual calcium transients elicited by trains of pre- and postsynaptic action potentials.
:
: drho/dt = -(1/τ)ρ(1-ρ)(ρ_star-ρ) + (γ_p/τ)(1-ρ) H[c(t)-θ_p] - (γ_d/τ)ρ H[c(t)-θ_d] + N
:       N = (σ/√τ) √(H[c(t)-θ_p] + H[c(t)-θ_d]) W
:
:   dc/dt = -(1/τ_Ca)c + C_pre Σ_i δ(t-t_i-D) + C_post Σ_j δ(t-t_j)
:
: rho      synaptic efficacy variable (unit-less)
: rho_star second root of cubic polynomial (unit-less), rho_star=0.5
: rho_0    initial value (unit-less)
: tau      synaptic time constant (ms), order of seconds to minutes
: gamma_p  rate of synaptic increase (unit-less)
: theta_p  potentiaton threshold (concentration)
: gamma_d  rate of synaptic decrease (unit-less)
: theta_d  depression threshold (concentration)
: sigma    noise amplitude
: W        white noise
: c        calcium concentration (concentration)
: C_pre    concentration jump after pre-synaptic spike (concentration)
: C_post   concentration jump after post-synaptic spike (concentration)
: tau_Ca   Calcium decay time constant (ms), order of milliseconds
: H        right-continuous heaviside step function ( H[x]=1 for x>=0; H[x]=0 otherwise )
: t_i      presynaptic spike times
: t_j      postsynaptic spike times
: D        time delay

NEURON {
    POINT_PROCESS calcium_based_synapse
    RANGE rho_0, tau, theta_p, gamma_p, theta_d, gamma_d, C_pre, C_post, tau_Ca, sigma
}

STATE {
    c
    rho
}

PARAMETER {
    rho_star = 0.5
    rho_0 = 1
    tau = 150000 (ms)
    gamma_p = 321.808
    theta_p = 1.3
    gamma_d = 200
    theta_d = 1
    sigma = 2.8248
    C_pre = 1
    C_post = 2
    tau_Ca = 20 (ms)
}

ASSIGNED {
    one_over_tau
    one_over_tau_Ca
    sigma_over_sqrt_tau
}

INITIAL {
    c = 0
    rho = rho_0

    one_over_tau = 1/tau
    one_over_tau_Ca = 1/tau_Ca
    sigma_over_sqrt_tau = sigma/(tau^0.5)
}

BREAKPOINT {
    SOLVE state METHOD stochastic
}

WHITE_NOISE {
    W
}

DERIVATIVE state {
    LOCAL hsp
    LOCAL hsd
    hsp = step_right(c - theta_p)
    hsd = step_right(c - theta_d)
    rho' = (-rho*(1-rho)*(rho_star-rho) + gamma_p*(1-rho)*hsp - gamma_d*rho*hsd)*one_over_tau + (hsp + hsd)^0.5*sigma_over_sqrt_tau*W
    c' = -c*one_over_tau_Ca
}

NET_RECEIVE(weight) {
    c = c + C_pre
}

POST_EVENT(time) {
    c = c + C_post
}
