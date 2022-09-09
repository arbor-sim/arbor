TITLE K-DR
: K-DR current for Mitral Cells from Wang et al (1996)
: M.Migliore Jan. 2002

: ek  (mV) must be explicitly def. in hoc
: ik  (mA/cm2)

NEURON {
    THREADSAFE
    SUFFIX kdrmt
    USEION k READ ek WRITE ik
    RANGE  gbar, vhalfm
}

PARAMETER {
    gbar   =  0.002    (mho/cm2)
    celsius
    a0m    =  0.0035
    vhalfm = -50
    zetam  =  0.055
    gmm    =  0.5
    q10    =  3
    alpm   =  0
    betm   =  0
}


UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
}

ASSIGNED { v(mV) }

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL gk
    gk = gbar*m
    ik = gk*(v - ek)
}

INITIAL {
    m = minf(v)
}

DERIVATIVE states {
    LOCAL qt, tmp

    qt  = q10^(0.1*(celsius - 24))
    tmp = zetam*(v - vhalfm)
    m'  = qt*a0m*(1 + exp(tmp))*(minf(v) - m)*exp(-gmm*tmp)
}

FUNCTION minf(v) { minf = 1/(1 + exp((21 - v)/10)) }
