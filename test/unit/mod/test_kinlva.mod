: Adaption of T-type calcium channel from Wang, X. J. et al. 1991;
: c.f. NMODL file in ModelDB:
: https://senselab.med.yale.edu/modeldb/showModel.cshtml?model=53893
:
: Note the temperature rate correction factors of 5 (for m) and 3
: (for h <-> s <-> d) have been applied to match the model described
: in the current-clamp experiments (see p. 842).

NEURON {
    SUFFIX test_kinlva
    USEION ca WRITE ica
    NONSPECIFIC_CURRENT il
}

UNITS {
    (mS) = (millisiemens)
    (mV) = (millivolts)
    (mA) = (millamp)
}

PARAMETER {
    gbar = 0.0002 (S/cm2)
    gl =  0.0001 (S/cm2)
    eca = 120 (mV)
    el = -65 (mV)
}

STATE {
    m h s d
}

BREAKPOINT {
    SOLVE m_state METHOD cnexp
    SOLVE dsh_state METHOD sparse
    ica = gbar*m^3*h*(v-eca)
    il = gl*(v-el)
}

FUNCTION minf(v) {
    minf = 1/(1+exp(-(v+63)/7.8))
}

FUNCTION K(v) {
    K = (0.25+exp((v+83.5)/6.3))^0.5-0.5
}

DERIVATIVE m_state {
    LOCAL taum, mi, m_q10
    m_q10 = 5
    mi = minf(v)
    taum = (1.7+exp(-(v+28.8)/13.5))*mi
    m' = m_q10*(mi - m)/taum
}

KINETIC dsh_state {
    LOCAL k, alpha1, beta1, alpha2, beta2, dsh_q10
    dsh_q10 = 3
    k = K(v)
    alpha1 = dsh_q10*exp(-(v+160.3)/17.8)
    beta1 = alpha1*k
    alpha2 = dsh_q10*(1+exp((v+37.4)/30))/240/(1+k)
    beta2 = alpha2*k

    ~ s <-> h (alpha1, beta1)
    ~ d <-> s (alpha2, beta2)
}

INITIAL {
    LOCAL k, vrest
    vrest = -65
    k = K(v)
    m = minf(vrest)
    h = 1/(1+k+k^2)
    d = h*k^2
    s = 1-h-d
}
