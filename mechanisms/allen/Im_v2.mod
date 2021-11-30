: Based on Im model of Vervaeke et al. (2006)

NEURON   {
   SUFFIX Im_v2
   USEION k READ ek WRITE ik
   RANGE gbar, ik
}

UNITS {
   (S) =  (siemens)
   (mV) = (millivolt)
   (mA) = (milliamp)
}

PARAMETER {
   celsius          (degC)     
   gbar = 0.00001   (S/cm2)
}

STATE {
   m
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   ik = gbar*m*(v - ek)
}

DERIVATIVE states {
    LOCAL qt, mAlpha, mBeta, mInf, mRat, iab
    
    qt = 2.3^((celsius - 30)/10)

    mAlpha = 0.007 * exp(( 6 *       0.4 * (v + 48))/26.12)
    mBeta  = 0.007 * exp((-6 * (1 - 0.4) * (v + 48))/26.12)

    iab    = 1/(mAlpha + mBeta)

    mInf = mAlpha*iab
    mRat = qt/(15 + iab)
    
    m' = (mInf - m)*mRat
}

INITIAL {
    LOCAL mAlpha, mBeta
    
    mAlpha = 0.007*exp(( 6 *      0.4  * (v + 48))/26.12)
    mBeta  = 0.007*exp((-6 * (1 - 0.4) * (v + 48))/26.12)

    m = mAlpha/(mAlpha + mBeta)
}

