NEURON {
    SUFFIX test_inilining
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    alpha = 2
    beta  = 4.5
    gamma = -15
    delta = -0.2
}

ASSIGNED {
    o1
    o2
}

STATE {
    s
}

BREAKPOINT {
    o1 = foo(alpha, beta)
    rates(delta)
    s = o1 * o2
}

FUNCTION foo(x, y) {
    LOCAL temp
    if (x == 3) {
        foo = 2 * y
    } else if (x == 4) {
        foo = y
    } else {
        temp = exp(y)
        foo = temp * x
    }
}

FUNCTION dip(q) {
    dip = 2*q
}

FUNCTION jab(x) {
    jab = x*dip(21)
}

FUNCTION bar(x) {
    bar = foo(x, x+2) * jab(42)
}

PROCEDURE rates(x)
{
    LOCAL  t0, t1, t2

    t0 = bar(s)
    t1 = exprelr(t0)
    t2 = foo(t1 + 2, 5)
    o2 = t2 + 4
}