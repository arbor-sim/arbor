: Comment:   The transient component of the K current
: Reference: Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients
:            Korngreen and Sakmann, J. Physiology, 2000

NEURON {
    SUFFIX K_T
    USEION k READ ek WRITE ik
    RANGE gbar
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

STATE {
    m
    h
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar*m*m*m*m*h*(v-ek)
}

DERIVATIVE states {
    LOCAL qt, mRat, hRat, hInf, mInf

    qt = 2.3^((celsius-21)/10)

    mInf =  1/(1 + exp(-(v + 47 - vshift)/29))
    hInf =  1/(1 + exp( (v + 66 - vshift)/10))
    mRat =  qt/(0.34 + mTauF*0.92*exp(-((v + 71 - vshift)/59)^2))
    hRat =  qt/(8    + hTauF*49  *exp(-((v + 73 - vshift)/23)^2))

    m' = (mInf - m)*mRat
    h' = (hInf - h)*hRat
}

INITIAL{
    m =  1/(1 + exp(-(v + 47 - vshift)/29))
    h =  1/(1 + exp( (v + 66 - vshift)/10))
}
