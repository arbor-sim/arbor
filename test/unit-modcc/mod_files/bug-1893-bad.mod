NEURON {
    POINT_PROCESS bug_1893
}

INITIAL {
    c = 0
    rho = 0
    theta_p = 0
}

STATE {
    c
    rho
    theta_p
}

PARAMETER {
    tau_c = 150 (ms)
}

BREAKPOINT {
    SOLVE state METHOD cnexp
}

DERIVATIVE state {
    c' = -c/tau_c
    rho' = (c - theta_p) > 0
}
