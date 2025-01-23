: Kv2-like channel
: Adapted from model implemented in Keren et al. 2005
: Adjusted parameters to be similar to guangxitoxin-sensitive current in mouse CA1 pyramids from Liu and Bean 2014

NEURON {
   SUFFIX Kv2like
   USEION k READ ek WRITE ik
   RANGE gbar, qt
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

ASSIGNED { qt }

STATE { m h1 h2 }

BREAKPOINT {
   SOLVE states METHOD cnexp
   LOCAL g
   g = 0.5*gbar*m*m*(h1 + h2)
   ik = g*(v - ek)
}

DERIVATIVE states {
   LOCAL mAlpha, mBeta, hInf, h1Rat, h2Rat

   mAlpha = m_alpha(v)
   mBeta  = m_beta(v)

   hInf  = h_inf(v)
   h1Rat = qt/( 360 + (2289.8 + 23.7*v)*exp(-((v + 75) / 48)^2))
   h2Rat = qt/(2350 + 1380*exp(-0.011*v) - 210*exp(-0.03*v))
   if (h2Rat < 0) { h2Rat = 1e-3 }
   
   m'  = 0.4*qt*(mAlpha - m*(mAlpha + mBeta))
   h1' = (hInf - h1)*h1Rat
   h2' = (hInf - h2)*h2Rat
}

INITIAL {
   LOCAL hInf, mAlpha, mBeta

   qt = 2.3^((celsius-21)/10)
   mAlpha = m_alpha(v)
   mBeta  = m_beta(v)
   hInf   = h_inf(v)

   m  = mAlpha/(mAlpha + mBeta)
   h1 = hInf
   h2 = hInf
}

FUNCTION vtrap(x, y) { vtrap = y*exprelr(x/y) }
FUNCTION h_inf(v)    { h_inf = 1/(1 + exp((v + 58) / 11)) }
FUNCTION m_alpha(v)  { m_alpha = 0.12*vtrap(43 - v, 11) }
FUNCTION m_beta(v)   { m_beta  = 0.02*exp(-(v + 1.27) / 120) }
