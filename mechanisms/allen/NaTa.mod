: Reference: Colbert and Pan 2002

NEURON {
    SUFFIX NaTa
    USEION na READ ena WRITE ina
    RANGE gbar, qt
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    v                   (mV)
    celsius             (degC)

    gbar    =   0.00001 (S/cm2)

    malphaF =   0.182
    mbetaF  =   0.124
    mvhalf  = -48       (mV)
    mk      =   6       (mV)

    halphaF =   0.015
    hbetaF  =   0.015
    hvhalf  = -69       (mV)
    hk      =   6       (mV)
}

ASSIGNED { qt }

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
	LOCAL g
    g = gbar*m*m*m*h
    ina = g*(v-ena)
}

DERIVATIVE states {
    LOCAL ma, mb, ha, hb
    ma = m_alpha(v)
    mb = m_beta(v)
    ha = h_alpha(v)
    hb = h_beta(v)
    m' = qt*(ma - m*(ma + mb))
    h' = qt*(ha - h*(ha + hb))
}

INITIAL {
  LOCAL ma, mb, ha, hb
  qt = 2.3^((celsius-23)/10)
  ma = m_alpha(v)
  ha = h_alpha(v)
  mb = m_beta(v)
  hb = h_beta(v)

  m = ma/(ma + mb)
  h = ha/(ha + hb)
}

FUNCTION vtrap(x, y) { vtrap = y*exprelr(x/y) }
FUNCTION m_alpha(v) { m_alpha = malphaF * vtrap(-(v - mvhalf), mk) }
FUNCTION m_beta(v)  { m_beta  = mbetaF  * vtrap(  v - mvhalf,  mk) }
: ng - adjusted this to match actual Colbert & Pan values for soma model
FUNCTION h_alpha(v) { h_alpha = halphaF * vtrap(  v - hvhalf,  hk) }
FUNCTION h_beta(v)  { h_beta  = hbetaF  * vtrap(-(v - hvhalf), hk) }
