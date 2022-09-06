NEURON {
    POINT_PROCESS exp2syn
    RANGE tau1, tau2, e
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
}

PARAMETER {
    tau1 = 0.5 (ms)
    tau2 = 2   (ms)
    e    = 0   (mV)
}

ASSIGNED { factor }

STATE { A B }

INITIAL {
    LOCAL tp
    A = 0
    B = 0
    tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
    factor = 1 / (-exp(-tp/tau1) + exp(-tp/tau2))
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = (B - A)*(v - e)
}

DERIVATIVE state {
    A' = -A/tau1
    B' = -B/tau2
}

NET_RECEIVE(weight) {
    A = A + weight*factor
    B = B + weight*factor
}

