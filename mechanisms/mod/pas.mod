UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

NEURON {
    SUFFIX pas
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

INITIAL {}

PARAMETER {
    g = .001 (S/cm2)
    e = -65  (mV) : we use -65 for the ball and stick model, instead of Neuron default of -70
}

ASSIGNED {
    v (mV)
}
FUNCTION bar() {
    bar = 4
}

FUNCTION foo(x) {
    foo = x*bar()
}

BREAKPOINT {
    LOCAL a
    a = 1
    if (v > 1) {
        if (exp(e)*bar() < 3) {
            a = foo(3)
        }
    }
    i = g*(v - e)*a
}
