: Based on Kd model of Foust et al. (2011)

NEURON {
   SUFFIX Kd
   USEION k READ ek WRITE ik
   RANGE gbar, ik
}

UNITS {
   (S)  = (siemens)
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
   ik = gbar*m*h*(v - ek)
}

DERIVATIVE states {
  LOCAL mInf, hInf
  mInf = 1 - 1/(1 + exp((v + 43)/8))
  hInf =     1/(1 + exp((v + 67)/7.3))
  m' = (mInf - m)
  h' = (hInf - h)/1500
}

INITIAL {
  m = 1 - 1/(1 + exp((v + 43)/8))
  h =     1/(1 + exp((v + 67)/7.3))
}

