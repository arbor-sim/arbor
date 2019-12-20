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

STATE {
    s0
    s1
    s2
}

BREAKPOINT {
    s1 = foo(alpha, beta)
    rates(delta)
    s0 = s1 * s2
}
FUNCTION iden(x) {
    iden = x
}

FUNCTION foo(x, y) {
    LOCAL temp
    if (x == 3) {
        foo = 2 * y
    } else if (x == 4) {
        foo = y * iden(6 + x)
    } else {
        temp = exp(y)
        foo = temp * x
    }
}

FUNCTION dip(q, p) {
    LOCAL temp
    temp = log(p)
    dip = 2*q*temp
}

FUNCTION jab(x, y) {
    jab = x*dip(21, x/y)
}

FUNCTION bar(x, y) {
    LOCAL p
    p = y/3
    bar = foo(x, x+2) * jab(p, y)
}

PROCEDURE rates(x)
{
    LOCAL  t0, t1, t2

    t0 = bar(s1, s2)
    t1 = exprelr(t0)
    t2 = foo(t1 + 2, 5)
    s2 = t2 + 4
}