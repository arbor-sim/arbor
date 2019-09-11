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
    o1 = foo (alpha, beta)
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

FUNCTION bar(x) {
    bar = foo(x, x + 2)
}

PROCEDURE rates(x)
{
    LOCAL  t0, t1, t2

    t0 = bar(23)
    t1 = exprelr(t0)
    t2 = foo(t1, 5)
    o2 = t2 + 4
}