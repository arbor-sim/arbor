: Reference: Adams et al. 1982 - M-currents and other potassium currents in bullfrog sympathetic neurones
: Comment:   corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON {
    SUFFIX dummy
    USEION k READ ek WRITE ik
    RANGE gImbar
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gImbar = 0.00001 (S/cm2)
}

STATE {
    m
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gImbar*m*(v - ek)
}

DERIVATIVE states {
    LOCAL qt, mAlpha, mBeta

    qt     = 2.3^((34-21)/10)
    mAlpha = m_alpha(v)
    mBeta  = m_beta(v)

    m'     = qt*(mAlpha - m*(mAlpha + mBeta))
}

INITIAL {
    LOCAL mAlpha, mBeta

    mAlpha = m_alpha(v)
    mBeta  = m_beta(v)

    m = mAlpha/(mAlpha + mBeta)
}

FUNCTION m_alpha(v) {
    m_alpha = 3.3e-3*exp( 2.5*0.04*(v + 35))
}
FUNCTION m_beta(v) {
    m_beta  = 3.3e-3*exp(-2.5*0.04*(v + 35))
}

