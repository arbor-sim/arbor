TITLE nax
: Na current for axon. No slow inact.
: M.Migliore Jul. 1997
: added sh to account for higher threshold M.Migliore, Apr.2002

NEURON {
    SUFFIX nax
    USEION na READ ena WRITE ina
    RANGE  gbar, sh
}

PARAMETER {
    sh   = 5    (mV)
    gbar = 0.010    (mho/cm2)

    tha  =  -30 (mV)        : v 1/2 for act 
    qa   = 7.2  (mV)        : act slope (4.5)
    Ra   = 0.4  (/ms)       : open (v)
    Rb   = 0.124 (/ms)      : close (v)

    thi1  = -45 (mV)        : v 1/2 for inact
    thi2  = -45 (mV)        : v 1/2 for inact
    qd   = 1.5  (mV)        : inact tau slope
    qg   = 1.5  (mV)
    mmin=0.02
    hmin=0.5
    q10=2
    Rg   = 0.01 (/ms)       : inact recov (v)
    Rd   = .03  (/ms)       : inact (v)

    thinf  = -50 (mV)       : inact inf slope
    qinf  = 4    (mV)       : inact inf slope

    celsius
}


UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
}

ASSIGNED {
    v (mV)
    thegna      (mho/cm2)
    minf
    hinf
    mtau (ms)
    htau (ms)
}

STATE {
    m
    h
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    thegna = gbar*m*m*m*h
    ina = thegna * (v - ena)
}

INITIAL {
    trates(v,sh,celsius)
    m=minf
    h=hinf
}

DERIVATIVE states {
    trates(v,sh,celsius)
    m' = (minf-m)/mtau
    h' = (hinf-h)/htau
}

PROCEDURE trates(vm,sh2,celsius) {
    LOCAL  a, b, qt
    qt=q10^((celsius-24)/10)
    a = trap0(vm,tha+sh2,Ra,qa)
    b = trap0(-vm,-tha-sh2,Rb,qa)
    mtau = max(1/(a+b)/qt, mmin)
    minf = a/(a+b)

    a = trap0(vm,thi1+sh2,Rd,qd)
    b = trap0(-vm,-thi2-sh2,Rg,qg)
    htau = max(1/(a+b)/qt, hmin)
    hinf = 1/(1+exp((vm-thinf-sh2)/qinf))
}

FUNCTION trap0(v,th,a,q) {
    : trap0 = a * (v - th) / (1 - exp(-(v - th)/q))
    trap0 = a*q*exprelr(-(v-th)/q)
}
