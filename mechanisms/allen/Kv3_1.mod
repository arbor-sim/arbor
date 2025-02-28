: Comment: Kv3-like potassium current

NEURON {
    SUFFIX Kv3_1
    USEION k READ ek WRITE ik
    RANGE gbar
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar   = 0.00001 (S/cm2)
    vshift = 0       (mV)
}

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL g
    g = gbar*m
    ik = g*(v - ek)
}

DERIVATIVE states {
    LOCAL mInf, mRat, vs
    vs = v - vshift
    mRat = 0.25*(1 + exp(-(vs + 46.560)/44.140))
    m' = (m_inf(vs) - m)*mRat
}

INITIAL {
    m = m_inf(v - vshift)
}

FUNCTION m_inf(v) { m_inf = 1/(1 + exp(-(v - 18.700)/9.700)) }
