TITLE K-A
: K-A current for Mitral Cells from Wang et al (1996)
: M.Migliore Jan. 2002

: ek must be explicitly set in hoc
: ik has units  (mA/cm2)

NEURON {
    THREADSAFE
    SUFFIX kamt
    USEION k READ ek WRITE ik
    RANGE  gbar, q10
    GLOBAL minf, mtau, hinf, htau
}

PARAMETER {
    gbar = 0.002    (mho/cm2)

    celsius
    a0m=0.04
    vhalfm=-45
    zetam=0.1
    gmm=0.75

    a0h=0.018
    vhalfh=-70
    zetah=0.2
    gmh=0.99

    sha=9.9
    shi=5.7
    q10=3
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
    hinf
    htau (ms)
}

STATE {
    m
    h
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar*m*h*(v - ek)
}

INITIAL {
    trates(v,celsius)
    m=minf
    h=hinf
}

DERIVATIVE states {
    trates(v,celsius)
    m' = (minf-m)/mtau
    h' = (hinf-h)/htau
}

PROCEDURE trates(v,celsius) {
    LOCAL qt
    qt=q10^((celsius-24)/10)

    minf = 1/(1 + exp(-(v-sha-7.6)/14))
    mtau = betm(v)/(qt*a0m*(1+alpm(v)))

    hinf = 1/(1 + exp((v-shi+47.4)/6))
    htau = beth(v)/(qt*a0h*(1+alph(v)))
}

FUNCTION alpm(v) {
    alpm = exp(zetam*(v-vhalfm))
}

FUNCTION betm(v) {
    betm = exp(zetam*gmm*(v-vhalfm))
}

FUNCTION alph(v) {
    alph = exp(zetah*(v-vhalfh))
}

FUNCTION beth(v) {
    beth = exp(zetah*gmh*(v-vhalfh))
}
