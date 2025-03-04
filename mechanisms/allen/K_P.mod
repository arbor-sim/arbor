: Comment:   The persistent component of the K current
: Reference: Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients
:            Korngreen and Sakmann
:            J. Physiology, 2000

NEURON {
    SUFFIX K_P
    USEION k READ ek WRITE ik
    RANGE gbar, vshift, tauF, qt
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    v                 (mV)
    celsius           (degC)
    gbar    = 0.00001 (S/cm2)
    vshift  = 0       (mV)
    tauF    = 1
}

STATE { m h }

ASSIGNED { qt }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL g
    g = gbar*m*m*h
    ik = g*(v - ek)
}

DERIVATIVE states {
    LOCAL m_rho, h_rho, vs, m_z

    vs = v - vshift

    if (vs < -50) {
        m_z = 175.03*exp( 0.026*vs)
    } else {
        m_z =  13.00*exp(-0.026*vs)
    }

    m_rho = qt/(tauF*(1.25 + m_z))
    h_rho = qt/(360 + (24*vs + 2330)*exp(-((vs + 75)/48)^2))
    m' = (m_inf(vs) - m)*m_rho
    h' = (h_inf(vs) - h)*h_rho
}

INITIAL{
    qt = 2.3^(0.1*(celsius - 21))
    m = m_inf(v - vshift)
    h = h_inf(v - vshift)
}

FUNCTION m_inf(vs) { m_inf = 1/(1 + exp(-(vs + 14.3)/14.6)) }
FUNCTION h_inf(vs) { h_inf = 1/(1 + exp( (vs + 54.0)/11)) }
