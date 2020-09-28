:Reference  Characterization of a Shaw-related potassium channel family in rat brain, The EMBO Journal, vol.11, no.7,2473-2486 (1992)

NEURON {
    SUFFIX SKv3_1
    USEION k READ ek WRITE ik
    RANGE gSKv3_1bar
}

UNITS    {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gSKv3_1bar = 0.00001 (S/cm2)
}

STATE {
    m
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gSKv3_1bar*m*(v - ek)
}

DERIVATIVE states {
    LOCAL mInf, mRho

    mInf = m_inf(v)
    mRho = 0.25*(1 + exp((v + 46.56)/(-44.140)))

    m' = (mInf - m)*mRho
}

INITIAL {
    m = m_inf(v)
}

FUNCTION m_inf(v) {
    m_inf = 1/(1 + exp((18.7 - v)/9.7))
}
