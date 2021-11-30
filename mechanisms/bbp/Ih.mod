: Reference:  Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON {
    SUFFIX Ih
    NONSPECIFIC_CURRENT ihcn
    RANGE gIhbar
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gIhbar =   0.00001 (S/cm2)
    ehcn   = -45.0     (mV)
}

STATE {
    m
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ihcn = gIhbar*m*(v - ehcn)
}

DERIVATIVE states {
    LOCAL mAlpha, mBeta

    mAlpha = m_alpha(v)
    mBeta  = m_beta(v)

    m' = mAlpha - m*(mAlpha + mBeta)
}

INITIAL {
    LOCAL mAlpha, mBeta

    mAlpha = m_alpha(v)
    mBeta  = m_beta(v)

    m = mAlpha/(mAlpha + mBeta)
}

FUNCTION m_alpha(v) {
    m_alpha = 0.001*6.43*11.9*exprelr((v + 154.9)/11.9)
}
FUNCTION m_beta(v) {
    m_beta  = 0.001*193*exp(v/33.1)
}

