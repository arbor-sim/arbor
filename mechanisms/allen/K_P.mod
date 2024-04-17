: Comment: The persistent component of the K current
: Reference:        Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients,Korngreen and Sakmann, J. Physiology, 2000


NEURON {
    SUFFIX K_P
    USEION k READ ek WRITE ik
    RANGE gbar, vshift, tauF, qt
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    v    (mV)
    celsius (degC)
    gbar = 0.00001 (S/cm2)
    vshift = 0 (mV)
    tauF = 1
}

STATE { m h }

ASSIGNED { qt }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL g
    g = gbar*m*m*h
    ik = g*(v-ek)
}

DERIVATIVE states {
    LOCAL m_rho, h_rho

    if (v < -50 + vshift){
        m_rho = qt/(tauF * (1.25 + 175.03*exp(-(v - vshift) * -0.026)))
       } else {
        m_rho = qt/(tauF * (1.25 +  13.00*exp(-(v - vshift) * 0.026)))
    }

    h_rho =  qt/(360 + (1010 + 24*(v + 55 - vshift))*exp(-((v + 75 - vshift)/48)^2))

    m' = qt*(m_inf(v) - m)*m_rho
    h' = qt*(h_inf(v) - h)*h_rho
}

INITIAL{
    qt = 2.3^(0.1*(celsius - 21))
    m = m_inf(v)
    h = h_inf(v)
}

FUNCTION m_inf(v) { m_inf = 1/(1 + exp(-(v + 14.3 - vshift)/14.6)) }
FUNCTION h_inf(v) { h_inf = 1/(1 + exp( (v + 54.0 - vshift)/11)) }
