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
    sh    = 5     (mV)
    gbar  = 0.010 (mho/cm2)

    tha   =  -30  (mV)  : v 1/2 for act
    qa    = 7.2   (mV)  : act slope (4.5)
    Ra    = 0.4   (/ms) : open (v)
    Rb    = 0.124 (/ms) : close (v)

    thi1  = -45   (mV)  : v 1/2 for inact
    thi2  = -45   (mV)  : v 1/2 for inact
    qd    = 1.5   (mV)  : inact tau slope
    qg    = 1.5   (mV)
    mmin  = 0.02
    hmin  = 0.5
    q10   = 2
    Rg    = 0.01  (/ms) : inact recov (v)
    Rd    = .03   (/ms) : inact (v)

    thinf = -50   (mV)  : inact inf slope
    qinf  = 4     (mV)  : inact inf slope

    celsius
}


UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
}

ASSIGNED { v (mV) }

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL gna
    gna = gbar*m*m*m*h
    ina = gna * (v - ena)
}

INITIAL {
    LOCAL  a, b, u

    u = tha + sh - v
    a = trap0( u, Ra, qa)
    b = trap0(-u, Rb, qa)
    m = a/(a + b)

    u = thi1 + sh - v
    a = trap0( u, Rd, qd)
    b = trap0(-u, Rg, qg)
    h = 1/(1 + exp((v - thinf - sh)/qinf))

}

DERIVATIVE states {
    LOCAL  a, b, u, qt, hinf, minf, htau, mtau, iq

    iq   = q10^(-0.1*(celsius - 24))

    u    = tha + sh - v
    a    = trap0( u, Ra, qa)
    b    = trap0(-u, Rb, qa)
    mtau = max(iq/(a + b), mmin)
    minf = a/(a + b)

    u    = thi1 + sh - v
    a    = trap0( u, Rd, qd)
    b    = trap0(-u, Rg, qg)
    htau = max(iq/(a + b), hmin)
    hinf = 1/(1 + exp((v - thinf - sh)/qinf))

    m' = (minf - m)/mtau
    h' = (hinf - h)/htau
}

FUNCTION trap0(v, a, q) { trap0 = a*q*exprelr(v/q) }
