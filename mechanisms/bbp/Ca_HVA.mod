: Reference: Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993

NEURON {
   SUFFIX Ca_HVA
   USEION ca READ eca WRITE ica
   RANGE gCa_HVAbar
}

UNITS {
   (S)  = (siemens)
   (mV) = (millivolt)
   (mA) = (milliamp)
}

PARAMETER {
   gCa_HVAbar = 0.00001 (S/cm2)
}

STATE {
   m
   h
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   ica = gCa_HVAbar*m*m*h*(v - eca)
}

DERIVATIVE states {
   LOCAL mAlpha, mBeta, hAlpha, hBeta

   mAlpha = m_alpha(v)
   mBeta  = m_beta(v)
   hAlpha = h_alpha(v)
   hBeta  = h_beta(v)

   m' = mAlpha - m*(mAlpha + mBeta)
   h' = hAlpha - h*(hAlpha + hBeta)
}

INITIAL {
   LOCAL mAlpha, mBeta, hAlpha, hBeta

   mAlpha = m_alpha(v)
   mBeta  = m_beta(v)
   hAlpha = h_alpha(v)
   hBeta  = h_beta(v)

   m = mAlpha/(mAlpha + mBeta)
   h = hAlpha/(hAlpha + hBeta)
}

FUNCTION m_alpha(v) {
   m_alpha = 0.055*3.8*exprelr(-(27 + v)/3.8)
}
FUNCTION m_beta(v) {
   m_beta  = 0.94*exp(-(v + 75)/17)
}
FUNCTION h_alpha(v) {
   h_alpha = 0.000457*exp(-(v + 13)/50)
}
FUNCTION h_beta(v) {
   h_beta  = 0.0065/(exp(-(v + 15)/28) + 1)
}
