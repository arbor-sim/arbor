NEURON  {
    SUFFIX NaTa_t
    USEION na READ ena WRITE ina
    RANGE gNaTa_tbar
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gNaTa_tbar = 0.00001 (S/cm2)
}

STATE {
    m
    h
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gNaTa_tbar*m*m*m*h*(v-ena)
}

DERIVATIVE states {
    LOCAL qt, mAlpha, mBeta, mRate, hAlpha, hBeta, hRate
    qt = 2.3^((34-21)/10)

    mAlpha = m_alpha(v)
    mBeta  = m_beta(v)
    mRate  = mAlpha + mBeta

    hAlpha = h_alpha(v)
    hBeta  = h_beta(v)
    hRate  = hAlpha + hBeta

    m' = qt*(mAlpha - m*mRate)
    h' = qt*(hAlpha - h*hRate)
}

INITIAL {
    LOCAL mAlpha, mBeta, hAlpha, hBeta

    mAlpha = m_alpha(v)
    mBeta  = m_beta(v)
    m      = mAlpha/(mAlpha + mBeta)

    hAlpha = h_alpha(v)
    hBeta  = h_beta(v)
    h      = hAlpha/(hAlpha + hBeta)
}

FUNCTION m_alpha(v) {
    m_alpha = 0.182*6*exprelr(-(v + 38)/6)
}

FUNCTION m_beta(v) {
    m_beta = 0.124*6*exprelr( (v + 38)/6)
}

FUNCTION h_alpha(v) {
    h_alpha = 0.015*6*exprelr( (v + 66)/6)
}

FUNCTION h_beta(v) {
    h_beta = 0.015*6*exprelr(-(v + 66)/6)
}
