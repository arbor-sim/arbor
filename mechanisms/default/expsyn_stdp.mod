: Exponential synapse with online STDP
: cf. https://brian2.readthedocs.io/en/stable/resources/tutorials/2-intro-to-brian-synapses.html#more-complex-synapse-models-stdp

NEURON {
    POINT_PROCESS expsyn_stdp
    RANGE tau, taupre, taupost, e, Apost, Apre, max_weight
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
}

PARAMETER {
    tau = 2.0 (ms) : synaptic time constant
    taupre = 10 (ms) : time constant of the pre-synaptic eligibility trace
    taupost = 10 (ms) : time constant of the post-synaptic eligibility trace
    Apre = 0.01 : pre-synaptic contribution
    Apost = -0.01  : post-synaptic contribution
    e = 0   (mV) : reversal potential
    max_weight = 10 (uS) : maximum synaptic conductance
}

STATE {
    g
    apre
    apost
    weight_plastic
}

INITIAL {
    g=0
    apre=0
    apost=0
    weight_plastic=0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = g*(v-e)
}

DERIVATIVE state {
    g' = -g/tau
    apre' = -apre/taupre
    apost' = -apost/taupost
}

NET_RECEIVE(weight) {
    g = max(0, min(g + weight + weight_plastic, max_weight))
    apre = apre + Apre
    weight_plastic = weight_plastic + apost
}

POST_EVENT(time) {
    apost = apost + Apost
    weight_plastic = weight_plastic + apre
}
