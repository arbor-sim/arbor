NEURON {
    SUFFIX Nap_Et2
    USEION na READ ena WRITE ina
    RANGE gNap_Et2bar
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gNap_Et2bar = 0.00001 (S/cm2)
}

STATE {
    m
    h
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gNap_Et2bar*m*m*m*h*(v - ena)
}

DERIVATIVE states {
    LOCAL qt, mInf, mAlpha, mBeta, mTau, hInf, hAlpha, hBeta, hTau

    qt = 2.3^((34 - 21)/10)

    mInf = m_inf(v)
    mAlpha = -0.182*6*exprelr(-(v + 38)/6)
    mBeta  = -0.124*6*exprelr( (v + 38)/6)
    mTau   = 6.0/(mAlpha + mBeta)

    hInf = h_inf(v)
    hAlpha = -2.88e-6*4.63*exprelr( (v + 17.0)/4.63)
    hBeta  = -6.94e-6*2.63*exprelr(-(v + 64.4)/2.63)
    hTau   = 1.0/(hAlpha + hBeta)

    m' = qt*(mInf - m)/mTau
    h' = qt*(hInf - h)/hTau
}

INITIAL {
    m = m_inf(v)
    h = h_inf(v)
}

FUNCTION m_inf(v) {
    m_inf = 1.0/(1 + exp(-(v + 52.6)/4.6))
}

FUNCTION h_inf(v) {
    h_inf = 1.0/(1 + exp( (v + 48.8)/10))
}
