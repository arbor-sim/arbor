: Based on Kd model of Foust et al. (2011)

NEURON {
   SUFFIX Kd
   USEION k READ ek WRITE ik
   RANGE gbar
}

UNITS {
   (S)  = (siemens)
   (mV) = (millivolt)
   (mA) = (milliamp)
}

PARAMETER {
   gbar = 0.00001 (S/cm2)
}

STATE { m h }

BREAKPOINT {
   SOLVE states METHOD cnexp
   LOCAL g
   g = gbar*m*h
   ik = g*(v - ek)
}

DERIVATIVE states {
  m' =  m_inf(v) - m
  h' = (h_inf(v) - h)/1500
}

INITIAL {
  m = m_inf(v)
  h = h_inf(v)
}

FUNCTION m_inf(v) { m_inf = 1 - 1/(1 + exp((v + 43)/8)) }
FUNCTION h_inf(v) { h_inf =     1/(1 + exp((v + 67)/7.3)) }
