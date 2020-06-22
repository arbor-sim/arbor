: Reference:        Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON {
    SUFFIX Ih
    NONSPECIFIC_CURRENT ihcn
    RANGE gbar
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0.00001 (S/cm2)
    ehcn =  -45.0 (mV)
}

STATE {
    m
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ihcn = gbar*m*(v-ehcn)
}

DERIVATIVE states {
   LOCAL mAlpha, mBeta, mRat
			       
    mAlpha = 0.001*6.43*vtrap(v + 154.9, 11.9)
    mBeta  = 0.001*193*exp(v/33.1)
    mRat   = mAlpha + mBeta

    m' = mAlpha - m*mRat
}

INITIAL {
    LOCAL mAlpha, mBeta

    mAlpha = 0.001 * 6.43 * vtrap(v + 154.9, 11.9)
    mBeta  =  0.001*193*exp(v/33.1)

    m = mAlpha/(mAlpha + mBeta)
}

FUNCTION vtrap(x, y) { : Traps for 0 in denominator of rate equations
    vtrap = y*exprelr(x/y)			       
}
