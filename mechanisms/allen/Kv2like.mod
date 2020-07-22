: Kv2-like channel
: Adapted from model implemented in Keren et al. 2005
: Adjusted parameters to be similar to guangxitoxin-sensitive current in mouse CA1 pyramids from Liu and Bean 2014

NEURON {
   SUFFIX Kv2like
   USEION k READ ek WRITE ik
   RANGE gbar
}

UNITS {
   (S)  = (siemens)
   (mV) = (millivolt)
   (mA) = (milliamp)
}

PARAMETER {
   celsius            (degC)
   gbar    = 0.00001 (S/cm2)
}

STATE {
   m
   h1
   h2
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   ik = 0.5*gbar*m*m*(h1 + h2)*(v - ek)
}

DERIVATIVE states {
   LOCAL qt, mAlpha, mBeta, hInf, h1Rat, h2Rat, mRat
   qt = 2.3^((celsius-21)/10)
   
   mAlpha = 0.12*vtrap(43 - v, 11)
   mBeta  = 0.02*exp(-(v + 1.27) / 120)
   mRat   = 0.4*qt*(mAlpha + mBeta)
   
   hInf  =  1/(1 + exp((v + 58) / 11))
   h1Rat = qt/( 360 + (1010 + 23.7*(v + 54))*exp(-((v + 75) / 48)^2))
   h2Rat = qt/(2350 + 1380*exp(-0.011*v) - 210*exp(-0.03*v))
   
   if (h2Rat < 0) {
      h2Rat = 1e-3
   }
   
   m'  = 0.4*qt*mAlpha - m*mRat
   h1' = (hInf - h1)*h1Rat
   h2' = (hInf - h2)*h2Rat
}

INITIAL {
   LOCAL hInf, mAlpha, mBeta

   mAlpha = 0.12*vtrap(43 - v, 11)
   mBeta  = 0.02*exp(-(v + 1.27) / 120)

   hInf = 1/(1 + exp((v + 58) / 11))
   
   m  = mAlpha/(mAlpha + mBeta)
   h1 = hInf
   h2 = hInf
}

FUNCTION vtrap(x, y) { : Traps for 0 in denominator of rate equations
    vtrap = y*exprelr(x/y)
}
