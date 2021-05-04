: Leaky-integrate and fire mechanism
: When crossing the spike threshold, the membrane is pulled
: towards the reset potential with a (possibly) high conductance
: for the duration of the refactory period.
: The spike detector threshold should match the threshold of the mechanism.

NEURON {
    SUFFIX lif
    NONSPECIFIC_CURRENT i
    RANGE g_reset, g_leak, e_reset, e_leak, e_thresh, tau_refrac
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

STATE {
    refractory_counter
}

INITIAL {
    refractory_counter = tau_refrac + 1 : start not refractory
}

PARAMETER {
    g_reset = 1000 (S/cm2) : conductance towards reset potential
    g_leak = 0.001 (S/cm2) : conductance towards leak potential

    e_reset = -80 (mV) : reset potential
    e_leak = -70  (mV) : leak potential
    e_thresh = -50 (mV) : spike threshold

    tau_refrac = 0.1 (ms) : refractory period
}

BREAKPOINT {
    SOLVE state METHOD cnexp

    LOCAL g
    LOCAL e

    : threshold crossed -> start refractory counter
    if (v > e_thresh) {
       refractory_counter = 0
    }

    : choose between leak and reset potential
    if (refractory_counter <= tau_refrac) {
       g = g_reset
       e = e_reset
    } else {
       g = g_leak
       e = e_leak
    }

    i = g*(v - e)
}

DERIVATIVE state {
    refractory_counter' = 1
}
