: Comment:   The transient component of the K current
: Reference: Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients
:            Korngreen and Sakmann,
:            J. Physiology, 2000

NEURON {
    SUFFIX K_T
    USEION k READ ek WRITE ik
    RANGE gbar, qt, vshift, mTauF, hTauF
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar   = 0.00001 (S/cm2)
    vshift = 0       (mV)
    mTauF  = 1.0
    hTauF  = 1.0
    celsius          (degC)
}

STATE { m h }

ASSIGNED { qt }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL g
    g = gbar*m*m*m*m*h
    ik = g*(v - ek)
}

DERIVATIVE states {
    LOCAL m_rho, h_rho, vs

    vs = v - vshift

    m_rho =  qt/(0.34 + mTauF* 0.92*exp(-((vs + 71)/59)^2))
    h_rho =  qt/(8    + hTauF*49.  *exp(-((vs + 73)/23)^2))

    m' = (m_inf(vs) - m)*m_rho
    h' = (h_inf(vs) - h)*h_rho
}

INITIAL {
    qt = 2.3^(0.1*(celsius - 21))
    m = m_inf(v - vshift)
    h = h_inf(v - vshift)
}

FUNCTION m_inf(vs) { m_inf = 1/(1 + exp(-(vs + 47)/29)) }
FUNCTION h_inf(vs) { h_inf = 1/(1 + exp( (vs + 66)/10)) }
