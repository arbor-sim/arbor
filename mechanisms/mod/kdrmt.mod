TITLE K-DR
: K-DR current for Mitral Cells from Wang et al (1996)
: M.Migliore Jan. 2002

: ek  (mV) must be explicitly def. in hoc
: ik  (mA/cm2)

NEURON {
    THREADSAFE
    SUFFIX kdrmt
    USEION k READ ek WRITE ik
    RANGE  gbar, q10, vhalfm
    GLOBAL minf, mtau
}

PARAMETER {
    gbar = 0.002    (mho/cm2)

    celsius
    a0m=0.0035
    vhalfm=-50
    zetam=0.055
    gmm=0.5
    q10=3
    alpm=0
    betm=0
}


UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
}

ASSIGNED {
    v    (mV)
    minf
    mtau (ms)
}

STATE {
    m
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar*m*(v - ek)
}

INITIAL {
    trates(v,celsius)
    m=minf
}

DERIVATIVE states {
    trates(v,celsius)
    m' = (minf-m)/mtau
}

PROCEDURE trates(v,celsius) {
    LOCAL qt
    LOCAL alpm, betm
    LOCAL tmp
    qt=q10^((celsius-24)/10)
    minf = 1/(1 + exp(-(v-21)/10))
    tmp = zetam*(v-vhalfm)
    alpm = exp(tmp)
    betm = exp(gmm*tmp)
    mtau = betm/(qt*a0m*(1+alpm))
}
