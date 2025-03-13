: Comment: LVA ca channel.
:          Note:
:          - mtau is an approximation from the plots
:          - shifted by -10 mv to correct for junction potential
:          - corrected rates using q10 = 2.3, target temperature 34, orginal 21
: Reference: Avery and Johnston 1996, tau from Randall 1997

NEURON {
    SUFFIX Ca_LVA
    USEION ca READ eca WRITE ica
    RANGE gbar, qt
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0.00001 (S/cm2)
    celsius
}

ASSIGNED { qt }

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL g
    g = gbar*m*m*h
    ica = g*(v - eca)
}

DERIVATIVE states {
    LOCAL mRat, hRat

    mRat = qt/(5  + 20/(1 + exp((v + 35)/5)))
    hRat = qt/(20 + 50/(1 + exp((v + 50)/7)))
     
    m' = (m_inf(v) - m)*mRat
    h' = (h_inf(v) - h)*hRat
}

INITIAL {
    qt = 2.3^((celsius-21)/10)
    m = m_inf(v)
    h = h_inf(v)
}

FUNCTION h_inf(v) { h_inf = 1/(1 + exp( (v + 90)/6.4)) }
FUNCTION m_inf(v) { m_inf = 1/(1 + exp(-(v + 40)/6)) }
