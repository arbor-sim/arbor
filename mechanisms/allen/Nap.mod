:Reference : Modeled according to kinetics derived from Magistretti & Alonso 1999
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON {
  SUFFIX Nap
  RANGE gbar, qt
  USEION na READ ena WRITE ina
}

UNITS {
  (S)  = (siemens)
  (mV) = (millivolt)
  (mA) = (milliamp)
}

PARAMETER {
  gbar    = 0.00001
  celsius           (degC)
  v                 (mV)
}

STATE { h }

ASSIGNED { qt }

INITIAL {
  qt = 2.3 ^ ((celsius - 21) / 10)
  h = h_inf(v)
}

BREAKPOINT {
  SOLVE states METHOD cnexp
  LOCAL g, m_inf
  m_inf = 1.0 / (1 + exp(-(v + 52.6)/4.6))
  g = gbar * m_inf * h
  ina = g * (v - ena)
}

DERIVATIVE states {
  LOCAL ha, hb
  ha = 2.88e-6 * vtrap(  v + 17.0,  4.63)
  hb = 6.94e-6 * vtrap(-(v + 64.4), 2.63)
  h' = (h_inf(v) - h) * (ha + hb) * qt
}

FUNCTION h_inf(v) { h_inf = 1.0 / (1 + exp( (v + 48.8)/10)) }
FUNCTION vtrap(x, y) { vtrap = y*exprelr(x/y) }
