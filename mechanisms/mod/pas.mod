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

BREAKPOINT {
    LOCAL xi, yi
    xi = 2
    xi = fu(xi)

:    yi = recurse1(2)
    if (f(ga(2))) {
        i = g*(v - e)
    }
}

FUNCTION fu(a) {
    fu = a/2
    if (a > 10) {
        fu = 3
    }
}

FUNCTION f(a) {
    f = 1
}

FUNCTION ga(b) {
    ga = 3
}

FUNCTION shadow(x) {
    LOCAL x
    x = 2
    shadow = 1
}

:FUNCTION recurse1(x) {
:    LOCAL y
:    if (x<2) {
:        recurse1 = x
:    }
:    else {
:        y = x/2
:        recurse1 = recurse2(y)
:    }
:}
:
:FUNCTION recurse2(x) {
:    LOCAL y
:    if (x<2) {
:        recurse2 = x
:    }
:    else {
:        y = x/5
:        recurse2 = recurse1(y)
:    }
:}