: Reference:        Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993

NEURON  {
    SUFFIX Ca_HVA
    USEION ca READ eca WRITE ica
    RANGE gbar, g, ica
}

UNITS   {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER   {
    gbar = 0.00001 (S/cm2)
}

ASSIGNED    {
    v   (mV)
    g   (S/cm2)
    mInf
    mTau
    mAlpha
    mBeta
    hInf
    hTau
    hAlpha
    hBeta
}

STATE   {
    m
    h
}

BREAKPOINT  {
    SOLVE states METHOD cnexp
    g = gbar*m*m*h
    ica = g*(v-eca)
}

DERIVATIVE states   {
    rates(v)
    m' = (mInf-m)/mTau
    h' = (hInf-h)/hTau
}

INITIAL{
    rates(v)
    m = mInf
    h = hInf
}

PROCEDURE rates(v){
    UNITSOFF

    mAlpha = 0.055 * vtrap(-27 - v, 3.8)
    mBeta  =  (0.94*exp((-75-v)/17))
    mInf = mAlpha/(mAlpha + mBeta)
    mTau = 1/(mAlpha + mBeta)
    hAlpha =  (0.000457*exp((-13-v)/50))
    hBeta  =  (0.0065/(exp((-v-15)/28)+1))
    hInf = hAlpha/(hAlpha + hBeta)
    hTau = 1/(hAlpha + hBeta)

    UNITSON
}

FUNCTION vtrap(x, y) {
    :vtrap = x / (exp(x / y) - 1)

    : use exprelr builtin
    vtrap = y*exprelr(x/y)
}
