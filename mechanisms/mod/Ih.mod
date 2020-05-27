: Reference:        Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON  {
    SUFFIX Ih
    NONSPECIFIC_CURRENT ihcn
    RANGE gbar, g
}

UNITS   {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER   {
    gbar = 0.00001 (S/cm2) 
    ehcn =  -45.0 (mV)
}

ASSIGNED    {
    v   (mV)
    g   (S/cm2)
    mInf
    mTau
    mAlpha
    mBeta
}

STATE   { 
    m
}

BREAKPOINT  {
    SOLVE states METHOD cnexp
    g = gbar*m
    ihcn = g*(v-ehcn)
}

DERIVATIVE states   {
    rates(v)
    m' = (mInf-m)/mTau
}

INITIAL{
    rates(v)
    m = mInf
}

PROCEDURE rates(v){
    mAlpha = 0.001 * 6.43 * vtrap(v + 154.9, 11.9)
    mBeta  =  0.001*193*exp(v/33.1)
    mInf = mAlpha/(mAlpha + mBeta)
    mTau = 1/(mAlpha + mBeta)
}

FUNCTION vtrap(x, y) {
    :vtrap = x / (exp(x / y) - 1)

    : use exprelr builtin
    vtrap = y*exprelr(x/y)
}
