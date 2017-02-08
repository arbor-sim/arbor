: sample file
? another style of comments

NEURON  {
    THREADSAFE
    SUFFIX test
    USEION ca WRITE ik READ ki, cai
    RANGE  gkbar, ik, ek, ki, cai
    GLOBAL minf, mtau, hinf, htau
}

STATE {
    h (nA)
    m r
}

UNITS   {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
    (molar) = (1/liter)
    (mM) = (milli/liter)
}

PARAMETER {
    cai
    gkbar   = 0.1  (mho/cm2)
    celsius
    ek      = -100 (mV)    : must be explicitly def. in hoc
    v       (mV)           ? another style of comment
    vhalfm  =-43   (mV)
    km      =8
    vhalfh  =-67   (mV)
    kh      =7.3
    q10     =2.3
}

ASSIGNED {
    ki      (mA/cm2)
    ik      (mA/cm2)
    minf    mtau (ms)
    hinf    htau (ms)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gkbar * m*h*(v-ek)
}

INITIAL {
    trates(v)
    m=minf
    h=hinf
}

PROCEDURE trates(v) {
    LOCAL qt
    qt=q10^((celsius-22)/10)
    minf=1-1/(1+exp((v-vhalfm)/km))
    hinf=1/(1+exp((v-vhalfh)/kh))

    if(minf<0) {
        foo1()
    }
    else if (hinf<0) {
        foo2()
    }
    else {
        foo3()
    }

    if(minf>=m) {
        foo3()
    }

    mtau = 0.6
    htau = 1500
}

PROCEDURE foo1() {}
PROCEDURE foo2() {}
PROCEDURE foo3() {}

: the 'states' in the definition is giving the derivative a name
: this name is then used in the SOLVE statement above
: should states be a procedure with special declaration syntax (takes no arguments by default)?

DERIVATIVE states {
    trates(v)
    m' = (minf-m)/mtau
    h' = (hinf-h)/htau
}

FUNCTION okcinf(Vm)  {
    LOCAL a
    LOCAL b
    a = 1.25*(10^8)*(cai)*(cai)
    okcinf = a/(a+b)
}
