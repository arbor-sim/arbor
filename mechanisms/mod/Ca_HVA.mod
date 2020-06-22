: Reference:        Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993

NEURON {
    SUFFIX Ca_HVA
    USEION ca READ eca WRITE ica
    RANGE gbar
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0.00001 (S/cm2)
}

STATE {
    m
    h
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ica = gbar*m*m*h*(v-eca)
}

DERIVATIVE states {
    LOCAL mAlpha, mBeta, hAlpha, hBeta, mRat, hRat

    mAlpha = 0.055*vtrap(-27 - v, 3.8)
    mBeta  = 0.94*exp((-75-v)/17)
    mRat   = mAlpha + mBeta					 
					 
    hAlpha = 0.000457*exp((-13-v)/50)
    hBeta  = 0.0065/(exp((-v-15)/28)+1)
    hRat = hAlpha + hBeta					      

    m' = mAlpha - m*mRat
    h' = hAlpha - h*hRat
}

INITIAL {
    LOCAL mAlpha, mBeta, hAlpha, hBeta
				       
    mAlpha = 0.055*vtrap(-27 - v, 3.8)
    mBeta  = 0.94*exp((-75 - v)/17)
    m      = mAlpha/(mAlpha + mBeta)

    hAlpha =  0.000457*exp((-13-v)/50)
    hBeta  =  0.0065/(exp((-v-15)/28) + 1)
    h = hAlpha/(hAlpha + hBeta)
}

FUNCTION vtrap(x, y) { : Traps for 0 in denominator of rate equations
    vtrap = y*exprelr(x/y)			       
}
